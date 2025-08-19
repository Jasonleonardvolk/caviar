# iRis-Status-Check.ps1
# Complete status check for iRis components
# Run this FIRST to see what's ready

Write-Host "`n=== iRIS STATUS CHECK - $(Get-Date -Format 'yyyy-MM-dd HH:mm') ===" -ForegroundColor Cyan
Write-Host "Checking all critical components..." -ForegroundColor Yellow

$status = @{
    "Core Files" = @{}
    "Services" = @{}
    "Dependencies" = @{}
    "Show Modes" = @{}
}

# Check core files
Write-Host "`n[CHECKING CORE FILES]" -ForegroundColor Magenta

$coreFiles = @{
    "HUD Component" = "tori_ui_svelte\src\lib\components\HolographicDisplay.svelte"
    "Hologram Route" = "tori_ui_svelte\src\routes\hologram\+page.svelte"
    "Show Controller" = "tori_ui_svelte\src\lib\show\ShowController.ts"
    "Show Index" = "tori_ui_svelte\src\lib\show\index.ts"
    "Package.json" = "tori_ui_svelte\package.json"
    "Vite Config" = "tori_ui_svelte\vite.config.js"
}

foreach ($name in $coreFiles.Keys) {
    $path = $coreFiles[$name]
    if (Test-Path $path) {
        Write-Host "  ✓ $name" -ForegroundColor Green
        $status["Core Files"][$name] = "OK"
    } else {
        Write-Host "  ✗ $name" -ForegroundColor Red
        $status["Core Files"][$name] = "MISSING"
    }
}

# Check show modes
Write-Host "`n[CHECKING SHOW MODES]" -ForegroundColor Magenta

$modes = @("particles", "portal", "anamorph", "glyphs", "penrose")
foreach ($mode in $modes) {
    $modePath = "tori_ui_svelte\src\lib\show\modes\$mode.ts"
    if (Test-Path $modePath) {
        Write-Host "  ✓ Mode: $mode" -ForegroundColor Green
        $status["Show Modes"][$mode] = "OK"
    } else {
        Write-Host "  ✗ Mode: $mode" -ForegroundColor Red
        $status["Show Modes"][$mode] = "MISSING"
    }
}

# Check if node_modules exists
Write-Host "`n[CHECKING DEPENDENCIES]" -ForegroundColor Magenta

if (Test-Path "tori_ui_svelte\node_modules") {
    Write-Host "  ✓ node_modules installed" -ForegroundColor Green
    $status["Dependencies"]["node_modules"] = "OK"
} else {
    Write-Host "  ✗ node_modules NOT installed" -ForegroundColor Red
    Write-Host "    Run: cd tori_ui_svelte && pnpm install" -ForegroundColor Yellow
    $status["Dependencies"]["node_modules"] = "NOT INSTALLED"
}

# Check for Penrose service
if (Test-Path "services\penrose\main.py") {
    Write-Host "  ✓ Penrose service found" -ForegroundColor Green
    $status["Dependencies"]["Penrose"] = "OK"
} else {
    Write-Host "  ⚠ Penrose service not found (optional)" -ForegroundColor Yellow
    $status["Dependencies"]["Penrose"] = "OPTIONAL"
}

# Check if services are running
Write-Host "`n[CHECKING RUNNING SERVICES]" -ForegroundColor Magenta

$ports = @{
    5173 = "Vite Dev Server"
    3000 = "SSR Production"
    7401 = "Penrose API"
}

foreach ($port in $ports.Keys) {
    $connection = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
    if ($connection.TcpTestSucceeded) {
        Write-Host "  ✓ Port $port : $($ports[$port]) - RUNNING" -ForegroundColor Green
        $status["Services"][$ports[$port]] = "RUNNING"
    } else {
        Write-Host "  ○ Port $port : $($ports[$port]) - NOT RUNNING" -ForegroundColor Gray
        $status["Services"][$ports[$port]] = "STOPPED"
    }
}

# Check Git status
Write-Host "`n[GIT STATUS]" -ForegroundColor Magenta
$gitChanges = git status --porcelain
if ($gitChanges) {
    $changeCount = ($gitChanges -split "`n").Count
    Write-Host "  ⚠ $changeCount uncommitted changes" -ForegroundColor Yellow
    Write-Host "    Run: .\p.ps1" -ForegroundColor Cyan
} else {
    Write-Host "  ✓ Working directory clean" -ForegroundColor Green
}

# RECOMMENDATIONS
Write-Host "`n=== RECOMMENDED ACTIONS ===" -ForegroundColor Cyan

$needsInstall = $status["Dependencies"]["node_modules"] -eq "NOT INSTALLED"
$devRunning = $status["Services"]["Vite Dev Server"] -eq "RUNNING"

if ($needsInstall) {
    Write-Host "`n1. Install dependencies:" -ForegroundColor Yellow
    Write-Host "   cd tori_ui_svelte" -ForegroundColor White
    Write-Host "   pnpm install" -ForegroundColor White
    Write-Host "   cd .." -ForegroundColor White
}

if (-not $devRunning) {
    Write-Host "`n2. Start iRis:" -ForegroundColor Yellow
    Write-Host "   .\Start-iRis-Now.ps1" -ForegroundColor White
} else {
    Write-Host "`n✓ iRis is RUNNING!" -ForegroundColor Green
    Write-Host "  Open: http://localhost:5173/hologram?show=wow" -ForegroundColor Cyan
    Write-Host "  Hotkeys: 1-5 (modes), 0 (cycle), B (boost), G (ghost)" -ForegroundColor Gray
}

# Show summary
$coreOK = ($status["Core Files"].Values | Where-Object {$_ -eq "OK"}).Count
$coreTotal = $status["Core Files"].Count
$modesOK = ($status["Show Modes"].Values | Where-Object {$_ -eq "OK"}).Count
$modesTotal = $status["Show Modes"].Count

Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
Write-Host "Core Files: $coreOK/$coreTotal ready" -ForegroundColor $(if ($coreOK -eq $coreTotal) {'Green'} else {'Yellow'})
Write-Host "Show Modes: $modesOK/$modesTotal ready" -ForegroundColor $(if ($modesOK -eq $modesTotal) {'Green'} else {'Yellow'})

if ($coreOK -eq $coreTotal -and $modesOK -eq $modesTotal -and -not $needsInstall) {
    Write-Host "`n🚀 iRis is READY TO LAUNCH!" -ForegroundColor Green
} else {
    Write-Host "`n⚠ Some setup needed (see above)" -ForegroundColor Yellow
}