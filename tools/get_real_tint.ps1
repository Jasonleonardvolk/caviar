# Get REAL Tint - Alternative sources
$tintDir = "C:\Users\jason\Desktop\tori\kha\tools\tint"
New-Item -ItemType Directory -Path $tintDir -Force | Out-Null

Write-Host "Getting REAL Tint binary..." -ForegroundColor Green

# Try alternative CDN/mirror sources
$sources = @(
    "https://storage.googleapis.com/chromium-dawn/tint-standalone-win64.exe",
    "https://chrome-infra-packages.appspot.com/dl/chromium/dawn/tint/windows-amd64/+/latest",
    "https://commondatastorage.googleapis.com/chromium-browser-snapshots/Win_x64/LAST_CHANGE"
)

$success = $false
foreach ($url in $sources) {
    try {
        Write-Host "Trying: $url"
        Invoke-WebRequest -Uri $url -OutFile "$tintDir\tint.exe" -UseBasicParsing -TimeoutSec 10
        if (Test-Path "$tintDir\tint.exe") {
            $fileSize = (Get-Item "$tintDir\tint.exe").Length
            if ($fileSize -gt 1000) {  # More than 1KB, probably real
                Write-Host "Downloaded from: $url" -ForegroundColor Green
                $success = $true
                break
            }
        }
    } catch {
        Write-Host "Failed: $_" -ForegroundColor Yellow
    }
}

if (!$success) {
    Write-Host ""
    Write-Host "All automated sources failed. Manual options:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Build from source (most reliable):" -ForegroundColor Cyan
    Write-Host "   git clone https://dawn.googlesource.com/dawn" 
    Write-Host "   cd dawn"
    Write-Host "   cmake -B build"
    Write-Host "   cmake --build build --target tint"
    Write-Host ""
    Write-Host "2. Get from a Chromium build:" -ForegroundColor Cyan
    Write-Host "   Download any recent Chromium build and extract tint.exe from it"
    Write-Host ""
    Write-Host "3. Use WSL to get Linux version and build Windows version:" -ForegroundColor Cyan
    Write-Host "   In WSL: apt-get install tint (then copy and analyze how it works)"
}

# Add to PATH properly
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($currentPath -notlike "*$tintDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$currentPath;$tintDir", "User")
    Write-Host "Added to PATH" -ForegroundColor Green
}
