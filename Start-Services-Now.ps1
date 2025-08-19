# Start-Services-Now.ps1
# Spins up Penrose + Vite (and/or SSR depending on your version)
param(
    [ValidateSet('dev', 'prod', 'both')]
    [string]$Mode = 'dev',
    [switch]$NoPenrose
)

Write-Host @"
╔════════════════════════════════════╗
║     iRis Services Launcher         ║
║     Starting backend + frontend    ║
╚════════════════════════════════════╝
"@ -ForegroundColor Cyan

Write-Host "`nMode: $Mode" -ForegroundColor Yellow
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm')" -ForegroundColor Gray

# Clean up any existing processes
Write-Host "`nCleaning up existing processes..." -ForegroundColor Gray
$ports = @{
    5173 = "Vite Dev"
    3000 = "SSR Prod"
    7401 = "Penrose"
}

foreach ($port in $ports.Keys) {
    $process = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | 
               Select-Object -ExpandProperty OwningProcess -Unique
    if ($process) {
        Stop-Process -Id $process -Force -ErrorAction SilentlyContinue
        Write-Host "  Stopped $($ports[$port]) on port $port" -ForegroundColor Gray
    }
}

Start-Sleep -Seconds 1

# Start Penrose service (backend)
if (-not $NoPenrose) {
    if (Test-Path "D:\Dev\kha\services\penrose\main.py") {
        Write-Host "`n[PENROSE SERVICE]" -ForegroundColor Magenta
        Write-Host "Starting Penrose on port 7401..." -ForegroundColor Cyan
        
        Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoExit", "-Command", @"
Write-Host 'PENROSE SERVICE' -ForegroundColor Magenta
cd 'D:\Dev\kha\services\penrose'
if (Test-Path .venv\Scripts\Activate.ps1) {
    .\.venv\Scripts\Activate.ps1
} elseif (Test-Path ..\..\venv\Scripts\Activate.ps1) {
    ..\..\.venv\Scripts\Activate.ps1
} elseif (Test-Path ..\..\venv_tori_prod\Scripts\Activate.ps1) {
    ..\..\venv_tori_prod\Scripts\Activate.ps1
}
python -m uvicorn main:app --host 0.0.0.0 --port 7401 --log-level info
"@
        Write-Host "  → API Docs: http://127.0.0.1:7401/docs" -ForegroundColor Gray
    } else {
        Write-Host "`n[PENROSE SERVICE]" -ForegroundColor Magenta
        Write-Host "⚠ Penrose not found at services\penrose - skipping" -ForegroundColor Yellow
    }
}

# Start Frontend (Vite and/or SSR)
if ($Mode -eq 'dev' -or $Mode -eq 'both') {
    Write-Host "`n[VITE DEV SERVER]" -ForegroundColor Magenta
    Write-Host "Starting Vite dev server on port 5173..." -ForegroundColor Cyan
    
    Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoExit", "-Command", @"
Write-Host 'VITE DEV SERVER' -ForegroundColor Magenta
cd 'D:\Dev\kha\tori_ui_svelte'
if (!(Test-Path node_modules)) {
    Write-Host 'Installing dependencies...' -ForegroundColor Yellow
    pnpm install
}
pnpm dev --host 0.0.0.0 --port 5173
"@
    Write-Host "  → Dev URL: http://localhost:5173/" -ForegroundColor Gray
}

if ($Mode -eq 'prod' -or $Mode -eq 'both') {
    Write-Host "`n[SSR PRODUCTION]" -ForegroundColor Magenta
    Write-Host "Building and starting SSR on port 3000..." -ForegroundColor Cyan
    
    Start-Process powershell -WindowStyle Minimized -ArgumentList "-NoExit", "-Command", @"
Write-Host 'SSR PRODUCTION SERVER' -ForegroundColor Magenta
cd 'D:\Dev\kha\tori_ui_svelte'
if (!(Test-Path node_modules)) {
    Write-Host 'Installing dependencies...' -ForegroundColor Yellow
    pnpm install
}
Write-Host 'Building production bundle...' -ForegroundColor Yellow
pnpm run build
Write-Host 'Starting SSR server...' -ForegroundColor Green
`$env:PORT = 3000
node build\index.js
"@
    Write-Host "  → Prod URL: http://localhost:3000/" -ForegroundColor Gray
}

# Wait and verify
Write-Host "`nWaiting for services to start..." -ForegroundColor Yellow
$checks = @()
if ($Mode -eq 'dev' -or $Mode -eq 'both') { $checks += 5173 }
if ($Mode -eq 'prod' -or $Mode -eq 'both') { $checks += 3000 }
if (-not $NoPenrose) { $checks += 7401 }

$maxWait = 30
$waited = 0

while ($waited -lt $maxWait) {
    Start-Sleep -Seconds 2
    $waited += 2
    
    $allUp = $true
    foreach ($port in $checks) {
        $conn = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
        if (-not $conn.TcpTestSucceeded) {
            $allUp = $false
            break
        }
    }
    
    if ($allUp) {
        Write-Host "`n✅ ALL SERVICES RUNNING!" -ForegroundColor Green
        break
    }
    Write-Host "." -NoNewline -ForegroundColor Gray
}

# Final status
Write-Host "`n`n=== SERVICE STATUS ===" -ForegroundColor Cyan
foreach ($port in $checks) {
    $conn = Test-NetConnection -ComputerName localhost -Port $port -WarningAction SilentlyContinue
    $status = if ($conn.TcpTestSucceeded) { "✓ RUNNING" } else { "✗ NOT RUNNING" }
    $color = if ($conn.TcpTestSucceeded) { "Green" } else { "Red" }
    
    $service = switch ($port) {
        5173 { "Vite Dev" }
        3000 { "SSR Prod" }
        7401 { "Penrose API" }
    }
    
    Write-Host "  $service (port $port): $status" -ForegroundColor $color
}

Write-Host "`n=== READY TO SHOW ===" -ForegroundColor Magenta
Write-Host "Run: .\Start-Show.ps1" -ForegroundColor Cyan
Write-Host "Or open manually:" -ForegroundColor Gray

if ($Mode -eq 'dev' -or $Mode -eq 'both') {
    Write-Host "  http://localhost:5173/hologram?show=wow" -ForegroundColor White
}
if ($Mode -eq 'prod' -or $Mode -eq 'both') {
    Write-Host "  http://localhost:3000/hologram?show=wow" -ForegroundColor White
}

Write-Host "`nTo stop all: .\Stop-iRis.ps1" -ForegroundColor Gray