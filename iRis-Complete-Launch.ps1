# iRis-Complete-Launch.ps1
# THE COMPLETE TWO-STEP LAUNCH SEQUENCE
param(
    [switch]$Production,
    [switch]$SkipPenrose
)

Clear-Host
Write-Host @"
    ╔══════════════════════════════════════╗
    ║          iRis COMPLETE LAUNCH        ║
    ║     Services + Show = One Command    ║
    ╚══════════════════════════════════════╝
"@ -ForegroundColor Cyan

Write-Host "`nStarting complete launch sequence..." -ForegroundColor Yellow
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# Step 1: Start all services
Write-Host "`n[STEP 1/2] Starting Services..." -ForegroundColor Magenta

$mode = if ($Production) { 'prod' } else { 'dev' }
$penroseFlag = if ($SkipPenrose) { '-NoPenrose' } else { '' }

& "D:\Dev\kha\Start-Services-Now.ps1" -Mode $mode $penroseFlag

# Wait a moment for services to stabilize
Write-Host "`nWaiting for services to stabilize..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# Step 2: Open the show
Write-Host "`n[STEP 2/2] Opening Show..." -ForegroundColor Magenta

& "D:\Dev\kha\Start-Show.ps1" -Target $mode

Write-Host "`n╔══════════════════════════════════════╗" -ForegroundColor Green
Write-Host   "║        iRis FULLY OPERATIONAL        ║" -ForegroundColor Green
Write-Host   "╚══════════════════════════════════════╝" -ForegroundColor Green

Write-Host "`nEverything is running! Enjoy the show!" -ForegroundColor Cyan