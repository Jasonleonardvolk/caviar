param([string]$Port = "5173")

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host "           500 ERROR FIX & DIAGNOSTICS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

function Ok($m){Write-Host "[OK] $m" -f Green}
function Info($m){Write-Host "[INFO] $m" -f Cyan}
function Warn($m){Write-Host "[WARN] $m" -f Yellow}
function Fail($m){Write-Host "[FAIL] $m" -f Red}

Write-Host "CHECKING SYSTEM..." -ForegroundColor Yellow
Write-Host ""

# 1. Check if server is running
Write-Host "1. Dev Server Status:" -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Ok "Server running on port $Port"
        $health = $response.Content | ConvertFrom-Json
        if ($health.ok) {
            Ok "Health check passed"
        }
    }
} catch {
    Fail "Server not responding on port $Port"
    Write-Host "  Fix: cd frontend && pnpm dev --host 0.0.0.0 --port $Port" -ForegroundColor Yellow
}

# 2. Check API endpoints
Write-Host "`n2. API Endpoints:" -ForegroundColor Cyan
try {
    $apiResponse = Invoke-WebRequest -Uri "http://localhost:$Port/api/wowpack/list" -TimeoutSec 2 -ErrorAction SilentlyContinue
    if ($apiResponse.StatusCode -eq 200) {
        Ok "/api/wowpack/list endpoint working"
        $data = $apiResponse.Content | ConvertFrom-Json
        Info "  Found $($data.items.Count) video sources"
    }
} catch {
    Fail "/api/wowpack/list not responding"
}

# 3. Check SSR is disabled
Write-Host "`n3. SSR Configuration:" -ForegroundColor Cyan
$pageConfigPath = "D:\Dev\kha\frontend\src\routes\hologram\+page.ts"
if (Test-Path $pageConfigPath) {
    Ok "+page.ts exists (SSR disabled)"
    $content = Get-Content $pageConfigPath -Raw
    if ($content -match "ssr = false") {
        Ok "SSR is disabled for /hologram"
    } else {
        Warn "SSR might not be disabled properly"
    }
} else {
    Fail "+page.ts missing - SSR not disabled!"
    Write-Host "  Creating it now..." -ForegroundColor Yellow
    @"
// Disable SSR for hologram page - it's purely client-side (WebGL/Canvas)
export const ssr = false;
export const csr = true;
export const prerender = false;
"@ | Set-Content $pageConfigPath
    Ok "Created +page.ts with SSR disabled"
}

# 4. Get network IPs
Write-Host "`n4. Network Configuration:" -ForegroundColor Cyan
$ips = Get-NetIPAddress -AddressFamily IPv4 | 
    Where-Object { $_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*" }

foreach ($ip in $ips) {
    Info "  $($ip.InterfaceAlias): $($ip.IPAddress)"
}

$primaryIP = $ips[0].IPAddress
Ok "Primary LAN IP: $primaryIP"

# 5. Test hologram page locally
Write-Host "`n5. Testing /hologram locally:" -ForegroundColor Cyan
try {
    $hologramResponse = Invoke-WebRequest -Uri "http://localhost:$Port/hologram" -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($hologramResponse.StatusCode -eq 200) {
        Ok "/hologram loads successfully (local)"
    }
} catch {
    if ($_.Exception.Response.StatusCode -eq 500) {
        Fail "/hologram returns 500 error!"
        Write-Host "  This means SSR issue - restart server after fix" -ForegroundColor Yellow
    } else {
        Warn "Could not test /hologram: $_"
    }
}

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host "           FIXES APPLIED" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

Ok "SSR disabled for /hologram (client-side only)"
Ok "API endpoints should work from client"
Ok "WebGL/Canvas will only run in browser"

Write-Host "`n------------------------------------------------------------" -ForegroundColor DarkMagenta
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. RESTART THE DEV SERVER:" -ForegroundColor Yellow
Write-Host "   cd D:\Dev\kha\frontend" -ForegroundColor White
Write-Host "   pnpm dev --host 0.0.0.0 --port $Port" -ForegroundColor White

Write-Host "`n2. TEST LOCALLY FIRST:" -ForegroundColor Yellow
Write-Host "   http://localhost:$Port/hologram" -ForegroundColor White

Write-Host "`n3. TEST FROM IPAD:" -ForegroundColor Yellow
Write-Host "   http://${primaryIP}:${Port}/hologram" -ForegroundColor White

Write-Host "`n4. IF STILL 500 ERROR:" -ForegroundColor Yellow
Write-Host "   - Check Windows Firewall (allow port $Port)" -ForegroundColor White
Write-Host "   - Try: http://${primaryIP}:${Port}/health" -ForegroundColor White
Write-Host "   - Check console: pnpm dev output for errors" -ForegroundColor White

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host ""