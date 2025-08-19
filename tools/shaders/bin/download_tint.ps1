# download_tint.ps1
# Downloads Tint (WGSL validator/compiler) from Dawn releases

$ErrorActionPreference = "Stop"

Write-Host "🔍 Downloading Tint for Windows..." -ForegroundColor Cyan

# Possible download URLs (update these as needed)
$urls = @(
    # Dawn/Tint releases (check for latest)
    "https://github.com/google/dawn/releases/latest/download/tint-windows-amd64.exe",
    "https://github.com/google/dawn/releases/download/chromium%2F6478/tint.exe",
    # Backup: gfx-rs wgpu releases often include Naga which can validate WGSL
    "https://github.com/gfx-rs/wgpu/releases/latest/download/wgpu-windows-x64.zip"
)

$destination = Join-Path $PSScriptRoot "tint.exe"

# Try Dawn/Chromium builds
Write-Host "Attempting to download from Dawn/Chromium..." -ForegroundColor Yellow
Write-Host "Note: You may need to manually download from:" -ForegroundColor Yellow
Write-Host "  https://dawn.googlesource.com/dawn/" -ForegroundColor White
Write-Host "  https://github.com/google/dawn/releases" -ForegroundColor White
Write-Host "  https://ci.chromium.org/p/dawn/builders/ci" -ForegroundColor White

# Create a simple downloader function
function Download-File {
    param($url, $output)
    try {
        Write-Host "  Trying: $url" -ForegroundColor Gray
        Invoke-WebRequest -Uri $url -OutFile $output -ErrorAction Stop
        return $true
    } catch {
        Write-Host "  ❌ Failed: $_" -ForegroundColor Red
        return $false
    }
}

$downloaded = $false
foreach ($url in $urls) {
    if (Download-File -url $url -output $destination) {
        $downloaded = $true
        break
    }
}

if ($downloaded) {
    Write-Host "✅ Downloaded to: $destination" -ForegroundColor Green
    
    # Test if it's executable
    try {
        & $destination --version
        Write-Host "✅ Tint is working!" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Downloaded but may not be the right executable" -ForegroundColor Yellow
    }
} else {
    Write-Host @"

❌ Automatic download failed. Please manually download Tint:

1. Go to: https://dawn.googlesource.com/dawn/
2. Click on 'Releases' or 'Tags'
3. Download the Windows binary (tint.exe or tint-windows-amd64.exe)
4. Place it in: $PSScriptRoot
5. Rename to: tint.exe

Alternative: Download Naga (WGSL validator) from:
  https://github.com/gfx-rs/wgpu/releases
  Look for: naga-cli or wgpu_validate

"@ -ForegroundColor Yellow
}

Write-Host @"

📝 Once tint.exe is in place, test with:
   .\tint.exe --version
   .\tint.exe --help

"@ -ForegroundColor Cyan
