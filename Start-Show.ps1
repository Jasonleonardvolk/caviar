# Start-Show.ps1
# Opens /hologram?show=wow with hotkey instructions
param(
    [ValidateSet('dev', 'prod', 'auto')]
    [string]$Target = 'auto',
    
    [ValidateSet('chrome', 'edge', 'firefox', 'default')]
    [string]$Browser = 'default',
    
    [switch]$NoInstructions
)

Write-Host @"

    ██╗    ██╗ ██████╗ ██╗    ██╗
    ██║    ██║██╔═══██╗██║    ██║
    ██║ █╗ ██║██║   ██║██║ █╗ ██║
    ██║███╗██║██║   ██║██║███╗██║
    ╚███╔███╔╝╚██████╔╝╚███╔███╔╝
     ╚══╝╚══╝  ╚═════╝  ╚══╝╚══╝ 
    HOLOGRAPHIC SHOW MODE

"@ -ForegroundColor Magenta

# Auto-detect which server is running
if ($Target -eq 'auto') {
    Write-Host "Auto-detecting running services..." -ForegroundColor Gray
    
    $dev = Test-NetConnection -ComputerName localhost -Port 5173 -WarningAction SilentlyContinue
    $prod = Test-NetConnection -ComputerName localhost -Port 3000 -WarningAction SilentlyContinue
    
    if ($dev.TcpTestSucceeded) {
        $Target = 'dev'
        Write-Host "  ✓ Found Vite dev server" -ForegroundColor Green
    } elseif ($prod.TcpTestSucceeded) {
        $Target = 'prod'
        Write-Host "  ✓ Found SSR production server" -ForegroundColor Green
    } else {
        Write-Host "  ✗ No services detected!" -ForegroundColor Red
        Write-Host "`nStart services first with: .\Start-Services-Now.ps1" -ForegroundColor Yellow
        exit 1
    }
}

# Build URL
$url = if ($Target -eq 'prod') {
    "http://localhost:3000/hologram?show=wow"
} else {
    "http://localhost:5173/hologram?show=wow"
}

Write-Host "`nOpening: $url" -ForegroundColor Cyan

# Show control instructions
if (-not $NoInstructions) {
    Write-Host "`n=== HOLOGRAM CONTROLS ===" -ForegroundColor Yellow
    Write-Host @"
    
    MODES (press number keys):
    [1] Particles  - Quantum particle field
    [2] Portal     - Dimensional gateway effect  
    [3] Anamorph   - Morphing geometries
    [4] Glyphs     - Matrix rain enhanced
    [5] Penrose    - Mathematical tessellation
    
    CONTROLS:
    [0] Cycle      - Auto-rotate through modes
    [B] Boost      - Increase brightness/intensity
    [G] Ghost      - Fade/trail effect
    [Space] Pause  - Pause animations
    [R] Reset      - Reset to defaults
    
    PERFORMANCE:
    [↑/↓] Speed    - Adjust animation speed
    [←/→] Density  - Adjust particle density
    [+/-] Quality  - Adjust render quality
    
"@ -ForegroundColor Gray
}

# Open in browser
switch ($Browser) {
    'chrome' {
        $chromePath = "${env:ProgramFiles}\Google\Chrome\Application\chrome.exe"
        if (Test-Path $chromePath) {
            Start-Process $chromePath -ArgumentList $url, "--new-window"
        } else {
            Start-Process $url
        }
    }
    'edge' {
        Start-Process "msedge.exe" -ArgumentList $url
    }
    'firefox' {
        $firefoxPath = "${env:ProgramFiles}\Mozilla Firefox\firefox.exe"
        if (Test-Path $firefoxPath) {
            Start-Process $firefoxPath -ArgumentList $url
        } else {
            Start-Process $url
        }
    }
    default {
        Start-Process $url
    }
}

Write-Host "`n✅ SHOW STARTED!" -ForegroundColor Green
Write-Host "URL: $url" -ForegroundColor Cyan

# Monitor mode (optional)
Write-Host "`n=== MONITORING ===" -ForegroundColor Yellow
Write-Host "The show is running. Press Ctrl+C to exit this monitor." -ForegroundColor Gray
Write-Host "Services will continue running in background." -ForegroundColor Gray

# Quick status check
Write-Host "`nService Status:" -ForegroundColor Cyan
$services = @(
    @{Port = 5173; Name = "Vite Dev"},
    @{Port = 3000; Name = "SSR Prod"},
    @{Port = 7401; Name = "Penrose API"}
)

foreach ($svc in $services) {
    $conn = Test-NetConnection -ComputerName localhost -Port $svc.Port -WarningAction SilentlyContinue
    if ($conn.TcpTestSucceeded) {
        Write-Host "  ✓ $($svc.Name) - Running" -ForegroundColor Green
    }
}

Write-Host "`nTips:" -ForegroundColor Magenta
Write-Host "  • For best performance, use Chrome or Edge" -ForegroundColor Gray
Write-Host "  • Press F11 for fullscreen experience" -ForegroundColor Gray
Write-Host "  • Check console (F12) for debug info" -ForegroundColor Gray
Write-Host "`nTo stop all services: .\Stop-iRis.ps1" -ForegroundColor Yellow