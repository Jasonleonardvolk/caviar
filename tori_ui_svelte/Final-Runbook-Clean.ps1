# Final-Runbook-Clean.ps1
# Exact implementation of the final runbook from the review
# Clean & deterministic deployment sequence

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     Final Runbook - Clean & Deterministic     " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Always from the package folder
Set-Location "D:\Dev\kha\tori_ui_svelte"
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Gray
Write-Host ""

# 0) Cache-bust so compiled endpoints are fresh
Write-Host "[0] Cache-busting for fresh compilation..." -ForegroundColor Yellow
if (Test-Path .\.svelte-kit) { 
    Write-Host "  Removing .svelte-kit..." -ForegroundColor Gray
    Remove-Item .\.svelte-kit -Recurse -Force 
}
if (Test-Path .\node_modules\.vite) { 
    Write-Host "  Removing node_modules/.vite..." -ForegroundColor Gray
    Remove-Item .\node_modules\.vite -Recurse -Force 
}
Write-Host "  Cache cleared!" -ForegroundColor Green

# 1) Build
Write-Host "`n[1] Building application..." -ForegroundColor Yellow
Write-Host "  Installing dependencies..." -ForegroundColor Gray
& pnpm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pnpm install failed" -ForegroundColor Red
    exit 1
}

Write-Host "  Building production bundle..." -ForegroundColor Gray
& pnpm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pnpm run build failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Build complete!" -ForegroundColor Green

# 2) Verify (mock)
Write-Host "`n[2] Running verification in mock mode..." -ForegroundColor Yellow
$env:IRIS_USE_MOCKS = "1"
Write-Host "  IRIS_USE_MOCKS = 1" -ForegroundColor Gray

& .\tools\release\Verify-EndToEnd.ps1 -Mode mock -Port 3000 -StartServer -StopOnExit
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Verification failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Verification passed!" -ForegroundColor Green

# 3) Ship (PM2) with explicit env propagation
Write-Host "`n[3] Shipping with PM2..." -ForegroundColor Yellow
$env:IRIS_USE_MOCKS = "1"
$env:PORT = "3000"
Write-Host "  IRIS_USE_MOCKS = 1" -ForegroundColor Gray
Write-Host "  PORT = 3000" -ForegroundColor Gray

& .\tools\release\Reset-And-Ship.ps1 -UsePM2
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Reset-And-Ship failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Deployment complete!" -ForegroundColor Green

# 4) Smoke test the endpoints
Write-Host "`n[4] Running smoke tests..." -ForegroundColor Yellow

# Give PM2 a moment to stabilize
Start-Sleep -Seconds 2

try {
    Write-Host "  Testing PDF stats endpoint..." -ForegroundColor Gray
    $pdfStats = Invoke-RestMethod http://127.0.0.1:3000/api/pdf/stats
    if ($pdfStats.note -eq 'mock') {
        Write-Host "    PDF stats: OK (mock confirmed)" -ForegroundColor Green
        Write-Host "    Response: $($pdfStats | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
    } else {
        Write-Host "    PDF stats: WARNING - Missing mock note" -ForegroundColor Yellow
    }
} catch {
    Write-Host "    PDF stats: FAILED - $_" -ForegroundColor Red
}

try {
    Write-Host "  Testing Memory state endpoint..." -ForegroundColor Gray
    $memoryState = Invoke-RestMethod http://127.0.0.1:3000/api/memory/state
    if ($memoryState.note -eq 'mock') {
        Write-Host "    Memory state: OK (mock confirmed)" -ForegroundColor Green
        Write-Host "    Response: $($memoryState | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
    } else {
        Write-Host "    Memory state: WARNING - Missing mock note" -ForegroundColor Yellow
    }
} catch {
    Write-Host "    Memory state: FAILED - $_" -ForegroundColor Red
}

# (optional) RC tag - commented out by default
Write-Host "`n[5] Optional: Create RC tag (skipped)" -ForegroundColor Gray
# & powershell .\tools\git\Git-Workflow.ps1 rc "1.0.0-rc.1" -Message "RC1 iRis UI (mock backend)" -Push

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "         Final Runbook Complete!                " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Application running at: http://127.0.0.1:3000" -ForegroundColor Cyan
Write-Host "Mode: Mock (IRIS_USE_MOCKS=1)" -ForegroundColor Yellow
Write-Host ""
Write-Host "PM2 Commands:" -ForegroundColor Yellow
Write-Host "  npx pm2 logs iris        # View logs" -ForegroundColor Gray
Write-Host "  npx pm2 restart iris     # Restart" -ForegroundColor Gray
Write-Host "  npx pm2 stop iris        # Stop" -ForegroundColor Gray
Write-Host "  npx pm2 status           # Check status" -ForegroundColor Gray
Write-Host ""
Write-Host "To commit these changes:" -ForegroundColor Yellow
Write-Host '  git add `' -ForegroundColor Gray
Write-Host '    .\tools\release\Reset-And-Ship.ps1 `' -ForegroundColor Gray
Write-Host '    .\src\routes\api\pdf\stats\+server.ts `' -ForegroundColor Gray
Write-Host '    .\src\routes\api\memory\state\+server.ts `' -ForegroundColor Gray
Write-Host '    .\src\routes\renderer\+page.server.ts' -ForegroundColor Gray
Write-Host ''
Write-Host '  git commit -m "ship: cache-bust + pm2 --update-env; api: compile-safe mocks; renderer: SSR mock guard"' -ForegroundColor Gray
Write-Host ""
Write-Host "Done." -ForegroundColor Green
