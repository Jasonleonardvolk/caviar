# Alternative: Get Tint from Chromium Tools
# Chromium includes Tint in their depot_tools

$tintDir = "C:\Users\jason\Desktop\tori\kha\tools\tint"
New-Item -ItemType Directory -Path $tintDir -Force | Out-Null

Write-Host "Attempting to get Tint from Chromium tools..." -ForegroundColor Green

# Option 1: Direct from Chromium snapshots
$chromiumBase = "https://commondatastorage.googleapis.com/chromium-browser-snapshots/Win_x64/"

try {
    # Get latest build number
    $latestBuild = Invoke-RestMethod -Uri "$chromiumBase/LAST_CHANGE"
    Write-Host "Latest Chromium build: $latestBuild"
    
    # Download mini_installer which contains tools
    $downloadUrl = "$chromiumBase/$latestBuild/chrome-win.zip"
    Write-Host "Downloading from: $downloadUrl"
    
    $zipPath = "$env:TEMP\chrome-win.zip"
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing
    
    # Extract and find tint
    Expand-Archive -Path $zipPath -DestinationPath "$env:TEMP\chrome-extract" -Force
    
    # Search for tint in the extracted files
    $tintExe = Get-ChildItem -Path "$env:TEMP\chrome-extract" -Filter "tint.exe" -Recurse -ErrorAction SilentlyContinue
    
    if ($tintExe) {
        Copy-Item $tintExe.FullName -Destination "$tintDir\tint.exe" -Force
        Write-Host "SUCCESS: Found and copied tint.exe" -ForegroundColor Green
    } else {
        Write-Host "Tint not found in Chromium build" -ForegroundColor Yellow
    }
    
    # Cleanup
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
    Remove-Item "$env:TEMP\chrome-extract" -Recurse -Force -ErrorAction SilentlyContinue
    
} catch {
    Write-Host "Chromium approach failed: $_" -ForegroundColor Red
}

# Option 2: Use depot_tools to get it
Write-Host ""
Write-Host "Alternative: Use depot_tools (Google's build tools)" -ForegroundColor Cyan
Write-Host "1. Download: https://storage.googleapis.com/chrome-infra/depot_tools.zip"
Write-Host "2. Extract to C:\depot_tools"
Write-Host "3. Run: C:\depot_tools\gclient"
Write-Host "4. Then fetch Dawn: fetch dawn"
Write-Host "5. Build with: gn gen out/Release && ninja -C out/Release tint"

# Test if it worked
if (Test-Path "$tintDir\tint.exe") {
    & "$tintDir\tint.exe" --version
    Write-Host "Tint is ready!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Still no tint.exe. Run the build script:" -ForegroundColor Yellow
    Write-Host "  C:\Users\jason\Desktop\tori\kha\tools\build_tint.bat" -ForegroundColor White
}
