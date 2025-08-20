# Run-Demo-Ngrok.ps1
# Ultimate demo launcher with ngrok tunneling for itori.io
param(
    [string]$Port = "5173",
    [switch]$SkipQR = $false
)

Write-Host "`n" -NoNewline
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "          ITORI.IO HOLOGRAM DEMO LAUNCHER" -ForegroundColor Cyan
Write-Host "            Powered by ngrok + WebGL2 Shaders" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

function Ok($m){Write-Host "[OK] $m" -f Green}
function Info($m){Write-Host "[INFO] $m" -f Cyan}
function Warn($m){Write-Host "[WARN] $m" -f Yellow}
function Fail($m){Write-Host "[FAIL] $m" -f Red}

# Check if ngrok is installed
$ngrokPath = Get-Command ngrok -ErrorAction SilentlyContinue
if (-not $ngrokPath) {
    Fail "ngrok not found in PATH!"
    Write-Host ""
    Write-Host "To install ngrok:" -ForegroundColor Yellow
    Write-Host "  1. Download from https://ngrok.com/download" -ForegroundColor White
    Write-Host "  2. Extract ngrok.exe to C:\tools\" -ForegroundColor White
    Write-Host "  3. Add C:\tools to your PATH" -ForegroundColor White
    Write-Host "  4. Run: ngrok config add-authtoken YOUR_TOKEN" -ForegroundColor White
    Write-Host ""
    exit 1
}

Ok "ngrok found at: $($ngrokPath.Source)"

# Kill any existing ngrok processes
Write-Host "`nCleaning up existing tunnels..." -ForegroundColor Yellow
$existingNgrok = Get-Process ngrok -ErrorAction SilentlyContinue
if ($existingNgrok) {
    $existingNgrok | Stop-Process -Force
    Info "Stopped existing ngrok process"
}

# Start dev server in background
Write-Host "`nStarting development server..." -ForegroundColor Yellow
$devServerJob = Start-Job -ScriptBlock {
    param($port)
    Set-Location "D:\Dev\kha\frontend"
    & pnpm dev --host 0.0.0.0 --port $port
} -ArgumentList $Port

Info "Dev server starting on port $Port (background job)"

# Wait for dev server to be ready
Write-Host "Waiting for server to be ready..." -NoNewline
$attempts = 0
$maxAttempts = 30
while ($attempts -lt $maxAttempts) {
    Start-Sleep -Seconds 1
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$Port" -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host " READY!" -ForegroundColor Green
            break
        }
    } catch {}
    Write-Host "." -NoNewline
    $attempts++
}

if ($attempts -eq $maxAttempts) {
    Fail "Dev server failed to start!"
    exit 1
}

# Start ngrok tunnel
Write-Host "`nLaunching ngrok tunnel..." -ForegroundColor Yellow
$ngrokProcess = Start-Process ngrok -ArgumentList "http $Port --log stdout" -PassThru -NoNewWindow -RedirectStandardOutput "$env:TEMP\ngrok_output.txt"

# Wait for ngrok to establish tunnel
Start-Sleep -Seconds 3

# Get ngrok public URL via API
try {
    $ngrokApi = Invoke-RestMethod -Uri "http://localhost:4040/api/tunnels" -ErrorAction Stop
    $publicUrl = $ngrokApi.tunnels[0].public_url
    
    if ($publicUrl) {
        # Convert to HTTPS if needed
        $publicUrl = $publicUrl -replace "^http://", "https://"
        $hologramUrl = "$publicUrl/hologram"
        $dashboardUrl = "$publicUrl/dashboard"
        
        Write-Host "`n============================================================" -ForegroundColor Magenta
        Write-Host "           DEMO URLS READY!" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Magenta
        Write-Host ""
        
        Ok "Public Hologram URL: $hologramUrl"
        Ok "Dashboard URL: $dashboardUrl"
        Ok "Local URL: http://localhost:$Port"
        
        # Generate QR code if not skipped
        if (-not $SkipQR) {
            Write-Host "`n------------------------------------------------------------" -ForegroundColor DarkMagenta
            Write-Host "QR CODE FOR INVESTORS (Scan with iPhone/iPad):" -ForegroundColor Cyan
            Write-Host "------------------------------------------------------------" -ForegroundColor DarkMagenta
            
            try {
                & npx qrcode-terminal $hologramUrl
                Write-Host ""
                Info "QR code generated for: $hologramUrl"
            } catch {
                Warn "npx not found - install Node.js for QR codes"
                Info "Manual URL: $hologramUrl"
            }
        }
        
        # Copy URL to clipboard
        $hologramUrl | Set-Clipboard
        Info "Hologram URL copied to clipboard!"
        
        Write-Host "`n============================================================" -ForegroundColor Magenta
        Write-Host "           DEMO TALKING POINTS:" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Magenta
        Write-Host ""
        
        Write-Host "1. MOBILE DEMO:" -ForegroundColor Yellow
        Info "  'Scan this QR code on your iPhone or iPad'"
        Info "  'This is our holographic rendering pipeline'"
        Info "  'Tap Fullscreen for immersive mode'"
        
        Write-Host "`n2. TECHNICAL HIGHLIGHTS:" -ForegroundColor Yellow
        Info "  'WebGL2 shaders with RGB diffraction'"
        Info "  'Real-time video processing at 60 FPS'"
        Info "  'Works natively on iOS 26 - no apps needed'"
        
        Write-Host "`n3. CONTENT SHOWCASE:" -ForegroundColor Yellow
        Info "  'Switch between HOLO FLUX, MACH LIGHTFIELD, KINETIC LOGO'"
        Info "  '5.56 GB ProRes masters processed in real-time'"
        Info "  'Multiple codec support: AV1, HDR10, SDR'"
        
        Write-Host "`n============================================================" -ForegroundColor Magenta
        Write-Host "           COMMANDS:" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Magenta
        Write-Host ""
        Write-Host "Press Ctrl+C to stop the demo server" -ForegroundColor White
        Write-Host ""
        
        # Keep script running
        Write-Host "Demo server running... Press Ctrl+C to stop" -ForegroundColor Green
        
        # Monitor the processes
        while ($true) {
            if (-not (Get-Process ngrok -ErrorAction SilentlyContinue)) {
                Warn "ngrok process stopped"
                break
            }
            Start-Sleep -Seconds 5
        }
        
    } else {
        Fail "Could not get public URL from ngrok"
    }
} catch {
    Fail "Failed to connect to ngrok API: $_"
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check if ngrok is running: Get-Process ngrok" -ForegroundColor White
    Write-Host "  2. Check ngrok dashboard: http://localhost:4040" -ForegroundColor White
    Write-Host "  3. Verify ngrok auth: ngrok config check" -ForegroundColor White
}

# Cleanup on exit
Write-Host "`nCleaning up..." -ForegroundColor Yellow
if ($devServerJob) {
    Stop-Job $devServerJob -Force
    Remove-Job $devServerJob -Force
}
if ($ngrokProcess -and -not $ngrokProcess.HasExited) {
    Stop-Process $ngrokProcess -Force
}

Write-Host "Demo stopped." -ForegroundColor Red