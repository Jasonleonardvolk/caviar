# Start-iRis-Now.ps1
# ONE COMMAND to launch iRis with all services
param(
    [switch]$Production,
    [switch]$SkipPenrose,
    [switch]$OpenBrowser
)

Write-Host "`n=== STARTING iRIS ===" -ForegroundColor Cyan
Write-Host "Launching all required services..." -ForegroundColor Yellow

# Kill any existing processes on our ports
Write-Host "`nCleaning up old processes..." -ForegroundColor Gray
$ports = @(5173, 3000, 7401)
foreach ($port in $ports) {
    $process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | 
               Select-Object -ExpandProperty OwningProcess -Unique
    if ($process) {
        Stop-Process -Id $process -Force -ErrorAction SilentlyContinue
        Write-Host "  Stopped process on port $port" -ForegroundColor Gray
    }
}

Start-Sleep -Seconds 1

# Start Penrose (optional)
if (-not $SkipPenrose -and (Test-Path "services\penrose\main.py")) {
    Write-Host "`nStarting Penrose service..." -ForegroundColor Cyan
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit", "-Command", @"
cd 'D:\Dev\kha\services\penrose'
if (Test-Path .venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
} elseif (Test-Path ..\..\venv\Scripts\Activate.ps1) {
    ..\..\.venv\Scripts\Activate.ps1
}
python -m uvicorn main:app --host 0.0.0.0 --port 7401 --log-level warning
"@
    Write-Host "  Penrose will be at: http://127.0.0.1:7401/docs" -ForegroundColor Gray
}

# Start Frontend
if ($Production) {
    Write-Host "`nStarting Production SSR..." -ForegroundColor Cyan
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit", "-Command", @"
cd 'D:\Dev\kha\tori_ui_svelte'
if (!(Test-Path node_modules)) { pnpm install }
pnpm run build
`$env:PORT = 3000
node build\index.js
"@
    $url = "http://localhost:3000/hologram?show=wow"
    Write-Host "  Production will be at: $url" -ForegroundColor Green
} else {
    Write-Host "`nStarting Dev Server..." -ForegroundColor Cyan
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit", "-Command", @"
cd 'D:\Dev\kha\tori_ui_svelte'
if (!(Test-Path node_modules)) { pnpm install }
pnpm dev --host --port 5173
"@
    $url = "http://localhost:5173/hologram?show=wow"
    Write-Host "  Dev server will be at: $url" -ForegroundColor Green
}

# Wait for services to start
Write-Host "`nWaiting for services to start..." -ForegroundColor Yellow
$maxWait = 30
$waited = 0
$targetPort = if ($Production) { 3000 } else { 5173 }

while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 2
    $waited += 2
    
    $connection = Test-NetConnection -ComputerName localhost -Port $targetPort -WarningAction SilentlyContinue
    if ($connection.TcpTestSucceeded) {
        Write-Host "`n✓ iRis is RUNNING!" -ForegroundColor Green
        break
    }
    Write-Host "." -NoNewline -ForegroundColor Gray
}

if ($waited -ge $maxWait) {
    Write-Host "`n⚠ Services taking longer than expected to start" -ForegroundColor Yellow
    Write-Host "Check the console windows for errors" -ForegroundColor Yellow
} else {
    Write-Host "`n=== iRIS READY ===" -ForegroundColor Green
    Write-Host "URL: $url" -ForegroundColor Cyan
    Write-Host "`nHotkeys:" -ForegroundColor Yellow
    Write-Host "  1-5 : Switch modes (particles, portal, anamorph, glyphs, penrose)" -ForegroundColor Gray
    Write-Host "  0   : Cycle through all modes" -ForegroundColor Gray
    Write-Host "  B   : Boost brightness" -ForegroundColor Gray
    Write-Host "  G   : Ghost fade effect" -ForegroundColor Gray
    
    if ($OpenBrowser -or (Read-Host "`nOpen in browser? (Y/n)") -ne 'n') {
        Start-Process $url
    }
}

Write-Host "`nTo stop all services, run: .\Stop-iRis.ps1" -ForegroundColor Gray
Write-Host "To check status, run: .\iRis-Status-Check.ps1" -ForegroundColor Gray