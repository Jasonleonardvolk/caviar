# Run-Demo-LAN.ps1
# Quick LAN demo launcher (no ngrok required)
param(
    [string]$Port = "5173"
)

Write-Host "`n" -NoNewline
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "          ITORI.IO HOLOGRAM - LAN DEMO" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

function Ok($m){Write-Host "[OK] $m" -f Green}
function Info($m){Write-Host "[INFO] $m" -f Cyan}
function Warn($m){Write-Host "[WARN] $m" -f Yellow}

# Find LAN IP addresses
Write-Host "Detecting network interfaces..." -ForegroundColor Yellow

$ips = Get-NetIPAddress -AddressFamily IPv4 | 
    Where-Object { $_.PrefixOrigin -eq "Dhcp" -or $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" } |
    Select-Object IPAddress, InterfaceAlias

if ($ips.Count -eq 0) {
    Warn "No LAN IP addresses found!"
    $lanIP = "localhost"
} else {
    Write-Host "`nAvailable network interfaces:" -ForegroundColor Cyan
    foreach ($ip in $ips) {
        Info "  $($ip.InterfaceAlias): $($ip.IPAddress)"
    }
    
    # Use the first one (usually Wi-Fi)
    $lanIP = $ips[0].IPAddress
    Ok "Using IP: $lanIP"
}

$hologramUrl = "http://${lanIP}:${Port}/hologram"
$dashboardUrl = "http://${lanIP}:${Port}/dashboard"
$localUrl = "http://localhost:${Port}"

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host "           DEMO URLS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

Ok "Hologram (LAN): $hologramUrl"
Ok "Dashboard (LAN): $dashboardUrl"
Ok "Local: $localUrl"

# Generate QR code
Write-Host "`n------------------------------------------------------------" -ForegroundColor DarkMagenta
Write-Host "QR CODE FOR LAN ACCESS:" -ForegroundColor Cyan
Write-Host "------------------------------------------------------------" -ForegroundColor DarkMagenta

try {
    & npx qrcode-terminal $hologramUrl
    Write-Host ""
    Info "QR code generated for LAN access"
} catch {
    Warn "npx not found - install Node.js for QR codes"
    Write-Host ""
    Write-Host "Manual URL for investors:" -ForegroundColor Yellow
    Write-Host "  $hologramUrl" -ForegroundColor White
}

# Copy to clipboard
$hologramUrl | Set-Clipboard
Info "Hologram URL copied to clipboard!"

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host "           STARTING DEV SERVER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

Info "Server will start on port $Port"
Info "Binding to all network interfaces (0.0.0.0)"
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start dev server
Set-Location "D:\Dev\kha\frontend"
& pnpm dev --host 0.0.0.0 --port $Port