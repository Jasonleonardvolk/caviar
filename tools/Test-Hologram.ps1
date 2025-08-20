param([string]$Port = "5173")

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host "           HOLOGRAM DEMO - QUICK TEST" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

function Ok($m){Write-Host "[OK] $m" -f Green}
function Info($m){Write-Host "[INFO] $m" -f Cyan}
function Test($url, $desc) {
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Ok "$desc - Status: 200 OK"
            return $true
        } else {
            Write-Host "[WARN] $desc - Status: $($response.StatusCode)" -f Yellow
            return $false
        }
    } catch {
        if ($_.Exception.Response.StatusCode -eq 500) {
            Write-Host "[FAIL] $desc - Status: 500 Internal Error" -f Red
        } else {
            Write-Host "[FAIL] $desc - Error: $_" -f Red
        }
        return $false
    }
}

# Wait for server to be ready
Write-Host "Waiting for server to be ready..." -NoNewline
$attempts = 0
$maxAttempts = 15
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
    Write-Host " TIMEOUT!" -ForegroundColor Red
    Write-Host "Server may still be starting. Try again in a few seconds." -ForegroundColor Yellow
}

Write-Host "`n------------------------------------------------------------" -ForegroundColor DarkMagenta
Write-Host "TESTING ENDPOINTS:" -ForegroundColor Cyan
Write-Host ""

# Test various endpoints
Test "http://localhost:$Port" "Home page"
Test "http://localhost:$Port/health" "Health check"
Test "http://localhost:$Port/api/wowpack/list" "WOW Pack API"
Test "http://localhost:$Port/hologram" "Hologram page"
Test "http://localhost:$Port/dashboard" "Dashboard"

# Get LAN IP
$lanIP = (Get-NetIPAddress -AddressFamily IPv4 | 
    Where-Object { $_.IPAddress -like "192.168.*" } | 
    Select-Object -First 1).IPAddress

Write-Host "`n------------------------------------------------------------" -ForegroundColor DarkMagenta
Write-Host "ACCESS URLS:" -ForegroundColor Cyan
Write-Host ""

Ok "Local: http://localhost:$Port/hologram"
Ok "LAN:   http://${lanIP}:${Port}/hologram"

Write-Host "`n------------------------------------------------------------" -ForegroundColor DarkMagenta
Write-Host "GENERATE QR CODE FOR IPAD:" -ForegroundColor Cyan
Write-Host ""

$hologramUrl = "http://${lanIP}:${Port}/hologram"
try {
    & npx qrcode-terminal $hologramUrl
    Info "QR code generated for: $hologramUrl"
} catch {
    Info "Manual URL for iPad: $hologramUrl"
}

# Copy to clipboard
$hologramUrl | Set-Clipboard
Info "URL copied to clipboard!"

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host "           TEST ON IPAD NOW!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""
Info "1. Scan the QR code above with iPad camera"
Info "2. Or manually enter: $hologramUrl"
Info "3. Tap 'Fullscreen' for immersive mode"
Info "4. Tap anywhere to toggle HUD"
Write-Host ""