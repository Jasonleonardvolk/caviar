# iRis-LAUNCH.ps1
# MASTER LAUNCHER - Does everything in order
param(
    [switch]$SkipChecks,
    [switch]$AutoFix,
    [switch]$Production
)

Clear-Host
Write-Host @"
    ██╗██████╗ ██╗███████╗
    ██║██╔══██╗██║██╔════╝
    ██║██████╔╝██║███████╗
    ██║██╔══██╗██║╚════██║
    ██║██║  ██║██║███████║
    ╚═╝╚═╝  ╚═╝╚═╝╚══════╝
    Holographic Display System
"@ -ForegroundColor Cyan

Write-Host "`n=== IRIS MASTER LAUNCHER ===" -ForegroundColor Magenta
Write-Host "Preparing to launch holographic display..." -ForegroundColor Yellow

# Step 1: Status check
if (-not $SkipChecks) {
    Write-Host "`n[1/4] Running status check..." -ForegroundColor Cyan
    & "D:\Dev\kha\iRis-Status-Check.ps1"
    
    if ($AutoFix) {
        Write-Host "`n[2/4] Running quick fix..." -ForegroundColor Cyan
        & "D:\Dev\kha\iRis-Quick-Fix.ps1"
    } else {
        Write-Host "`n[2/4] Skipping quick fix (use -AutoFix to enable)" -ForegroundColor Gray
    }
} else {
    Write-Host "`n[1/4] Skipping checks (use -SkipChecks:$false for full check)" -ForegroundColor Gray
    Write-Host "[2/4] Skipping fixes" -ForegroundColor Gray
}

# Step 3: Check dependencies
Write-Host "`n[3/4] Checking dependencies..." -ForegroundColor Cyan
if (!(Test-Path "D:\Dev\kha\tori_ui_svelte\node_modules")) {
    Write-Host "  Installing dependencies..." -ForegroundColor Yellow
    Set-Location "D:\Dev\kha\tori_ui_svelte"
    & pnpm install
    Set-Location "D:\Dev\kha"
} else {
    Write-Host "  ✓ Dependencies already installed" -ForegroundColor Green
}

# Step 4: Launch
Write-Host "`n[4/4] Launching iRis..." -ForegroundColor Cyan
if ($Production) {
    & "D:\Dev\kha\Start-iRis-Now.ps1" -Production -OpenBrowser
} else {
    & "D:\Dev\kha\Start-iRis-Now.ps1" -OpenBrowser
}

Write-Host "`n=== LAUNCH SEQUENCE COMPLETE ===" -ForegroundColor Green