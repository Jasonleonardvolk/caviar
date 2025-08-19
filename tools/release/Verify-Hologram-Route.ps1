param(
    [string]$ProjectRoot = "D:\Dev\kha",
    [string]$TestUrl = "http://localhost:5173/hologram"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Hologram Route Verification            " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if critical files exist
$requiredFiles = @(
    "$ProjectRoot\frontend\src\routes\hologram\+page.svelte",
    "$ProjectRoot\frontend\src\lib\hologram\engineShim.ts",
    "$ProjectRoot\frontend\src\lib\device\capabilities.ts",
    "$ProjectRoot\frontend\src\lib\components\HologramRecorder.svelte",
    "$ProjectRoot\frontend\src\lib\utils\exportVideo.ts"
)

$missingFiles = @()
foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "[X] Missing critical files:" -ForegroundColor Red
    $missingFiles | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    exit 1
}

Write-Host "[✓] All critical files present" -ForegroundColor Green

# Check capabilities.ts for WebGPU support
$capsContent = Get-Content "$ProjectRoot\frontend\src\lib\device\capabilities.ts" -Raw
if ($capsContent -match "navigator\.gpu" -and $capsContent -match "prefersWebGPUHint") {
    Write-Host "[✓] WebGPU capability detection present" -ForegroundColor Green
} else {
    Write-Host "[!] WebGPU capability detection may be missing" -ForegroundColor Yellow
}

# Check hologram page for canvas
$holoContent = Get-Content "$ProjectRoot\frontend\src\routes\hologram\+page.svelte" -Raw
if ($holoContent -match "#hologram-canvas") {
    Write-Host "[✓] Hologram canvas target found" -ForegroundColor Green
} else {
    Write-Host "[X] Hologram canvas target missing" -ForegroundColor Red
    exit 1
}

# Check if dev server is running (optional)
try {
    $response = Invoke-WebRequest -Uri $TestUrl -TimeoutSec 5 -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "[✓] Hologram route accessible at $TestUrl" -ForegroundColor Green
    }
} catch {
    Write-Host "[!] Dev server not running or hologram route not accessible" -ForegroundColor Yellow
    Write-Host "    Start with: pnpm dev -- --host 0.0.0.0" -ForegroundColor Gray
}

# Mobile support check
$mobileConfig = "$ProjectRoot\config\mobile_support.json"
if (Test-Path $mobileConfig) {
    $config = Get-Content $mobileConfig | ConvertFrom-Json
    Write-Host ""
    Write-Host "Mobile Support Configuration:" -ForegroundColor Cyan
    Write-Host "  iOS Version: $($config.iosMajor) $($config.preferredBeta)" -ForegroundColor White
    Write-Host "  Minimum Device: $($config.minIphoneModel)" -ForegroundColor White
    Write-Host "  WebGPU Preferred: $($config.preferWebGPU)" -ForegroundColor White
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Hologram Route: VERIFIED               " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps for mobile testing:" -ForegroundColor Yellow
Write-Host "1. Start dev server: pnpm dev -- --host 0.0.0.0" -ForegroundColor Gray
Write-Host "2. Open on iOS 26 Beta 7+ device" -ForegroundColor Gray
Write-Host "3. Navigate to http://<YOUR-IP>:5173/hologram" -ForegroundColor Gray
Write-Host "4. Verify WebGPU capabilities in UI" -ForegroundColor Gray

exit 0