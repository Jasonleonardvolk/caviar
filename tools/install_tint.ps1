# Quick Tint Installation Script
# 2-minute approach, not 2-hour approach

$ErrorActionPreference = "Stop"

Write-Host "Installing Tint shader translator..." -ForegroundColor Green

# Create tools directory if not exists
$toolsDir = "C:\Users\jason\Desktop\tori\kha\tools\tint"
if (!(Test-Path $toolsDir)) {
    New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
}

# Download latest Tint (part of Dawn tools)
# Using a known working release
$url = "https://github.com/google/dawn/releases/download/chromium%2F6854/dawn-6854-windows-amd64.zip"
$zipPath = "$toolsDir\dawn.zip"

Write-Host "Downloading Tint from Dawn project..."
try {
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
} catch {
    Write-Host "Failed to download from primary URL, trying backup..." -ForegroundColor Yellow
    # Backup: build from source (this is the 2-hour path we want to avoid)
    Write-Host "ERROR: Auto-download failed. Manual steps:" -ForegroundColor Red
    Write-Host "1. Go to: https://github.com/google/dawn/releases"
    Write-Host "2. Download the latest Windows release"
    Write-Host "3. Extract tint.exe to: $toolsDir"
    Write-Host "4. Run: setx PATH `"%PATH%;$toolsDir`""
    exit 1
}

Write-Host "Extracting..."
Expand-Archive -Path $zipPath -DestinationPath $toolsDir -Force

# Find tint.exe in the extracted files
$tintExe = Get-ChildItem -Path $toolsDir -Filter "tint.exe" -Recurse | Select-Object -First 1

if ($tintExe) {
    # Move tint.exe to tools/tint root
    Move-Item -Path $tintExe.FullName -Destination "$toolsDir\tint.exe" -Force
    Write-Host "Tint installed to: $toolsDir\tint.exe" -ForegroundColor Green
} else {
    Write-Host "WARNING: tint.exe not found in archive. Check manually." -ForegroundColor Yellow
}

# Clean up
Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $toolsDir -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Add to PATH for current session
$env:Path += ";$toolsDir"

# Test if it works
try {
    & "$toolsDir\tint.exe" --version
    Write-Host "SUCCESS: Tint is working!" -ForegroundColor Green
} catch {
    Write-Host "Tint installed but not responding. Check manually." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "To add Tint permanently to PATH, run:" -ForegroundColor Cyan
Write-Host "  setx PATH `"%PATH%;$toolsDir`"" -ForegroundColor White
Write-Host ""
Write-Host "Or for system-wide (run as admin):" -ForegroundColor Cyan
Write-Host "  [Environment]::SetEnvironmentVariable('Path', `$env:Path + ';$toolsDir', 'Machine')" -ForegroundColor White
