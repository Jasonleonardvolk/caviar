param(
  [string]$RepoRoot = "D:\Dev\kha"
)

Set-Location $RepoRoot

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "TYPESCRIPT IMPORT FIX" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$realGhostPath = Join-Path $RepoRoot "tori_ui_svelte\src\lib\realGhostEngine.js"
$backupPath = Join-Path $RepoRoot "tori_ui_svelte\src\lib\realGhostEngine.js.backup"
$tempPath = Join-Path $RepoRoot "tori_ui_svelte\src\lib\realGhostEngine_temp.js"

Write-Host "`nThe problem: JavaScript can't import TypeScript files directly!" -ForegroundColor Yellow
Write-Host "TypeScript files need to be compiled to JavaScript first." -ForegroundColor Yellow

Write-Host "`nOptions:" -ForegroundColor Cyan
Write-Host "1. Use temporary mock imports (quick fix)" -ForegroundColor White
Write-Host "2. Compile TypeScript files first (proper fix)" -ForegroundColor White
Write-Host "3. Convert imports to JavaScript files" -ForegroundColor White

$choice = Read-Host "`nChoose option (1-3)"

switch ($choice) {
    "1" {
        Write-Host "`nApplying temporary mock fix..." -ForegroundColor Yellow
        
        # Backup original
        if (Test-Path $realGhostPath) {
            Copy-Item $realGhostPath $backupPath -Force
            Write-Host "  Backed up original to realGhostEngine.js.backup" -ForegroundColor Gray
        }
        
        # Use the temp file with mocks
        if (Test-Path $tempPath) {
            Copy-Item $tempPath $realGhostPath -Force
            Write-Host "  Applied mock imports" -ForegroundColor Green
        }
        
        Write-Host "`nTrying build with mocks..." -ForegroundColor Yellow
        npm run build
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "`nBUILD SUCCESSFUL with mocks!" -ForegroundColor Green
        } else {
            Write-Host "`nBuild still failed. Check for other issues." -ForegroundColor Red
        }
    }
    
    "2" {
        Write-Host "`nCompiling TypeScript files..." -ForegroundColor Yellow
        Write-Host "  This requires a proper TypeScript build setup" -ForegroundColor Gray
        
        # Try to compile frontend TypeScript
        $frontendPath = Join-Path $RepoRoot "frontend"
        if (Test-Path (Join-Path $frontendPath "tsconfig.json")) {
            Set-Location $frontendPath
            Write-Host "  Running tsc in frontend directory..." -ForegroundColor Gray
            npx tsc
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  TypeScript compiled!" -ForegroundColor Green
            } else {
                Write-Host "  TypeScript compilation failed" -ForegroundColor Red
            }
            Set-Location $RepoRoot
        }
    }
    
    "3" {
        Write-Host "`nTo convert to JavaScript imports:" -ForegroundColor Yellow
        Write-Host "  1. Find JavaScript versions of these files:" -ForegroundColor Gray
        Write-Host "     - holographicEngine.js" -ForegroundColor White
        Write-Host "     - holographicRenderer.js" -ForegroundColor White
        Write-Host "     - fftCompute.js" -ForegroundColor White
        Write-Host "     - hologramPropagation.js" -ForegroundColor White
        Write-Host "     - QuiltGenerator.js" -ForegroundColor White
        Write-Host "  2. Update import paths in realGhostEngine.js" -ForegroundColor Gray
    }
    
    default {
        Write-Host "`nNo action taken" -ForegroundColor Yellow
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "To revert changes:" -ForegroundColor Yellow
Write-Host "  Copy-Item '$backupPath' '$realGhostPath' -Force" -ForegroundColor White

exit 0