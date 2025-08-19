# Install-FFmpeg.ps1
# Downloads and installs FFmpeg to the tools directory

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "       FFmpeg Installation Script               " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$ffmpegDir = "D:\Dev\kha\tools\ffmpeg"
$ffmpegExe = Join-Path $ffmpegDir "ffmpeg.exe"
$ffprobeExe = Join-Path $ffmpegDir "ffprobe.exe"

# Check if already installed
if ((Test-Path $ffmpegExe) -and (Test-Path $ffprobeExe)) {
    Write-Host "FFmpeg already installed at: $ffmpegDir" -ForegroundColor Green
    
    # Add to current session PATH
    if ($env:Path -notlike "*$ffmpegDir*") {
        $env:Path = "$ffmpegDir;$env:Path"
        Write-Host "Added to current session PATH" -ForegroundColor Green
    }
    
    Write-Host ""
    Write-Host "To make permanent, run:" -ForegroundColor Yellow
    Write-Host '  [Environment]::SetEnvironmentVariable("Path", "$env:Path", [System.EnvironmentVariableTarget]::User)' -ForegroundColor Gray
    exit 0
}

Write-Host "FFmpeg not found. Installing..." -ForegroundColor Yellow

# Create directory
New-Item -ItemType Directory -Path $ffmpegDir -Force | Out-Null

# Download URL for Windows FFmpeg (latest stable)
$downloadUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$zipPath = Join-Path $ffmpegDir "ffmpeg.zip"

Write-Host "Downloading FFmpeg..." -ForegroundColor Yellow
Write-Host "  From: $downloadUrl" -ForegroundColor Gray

try {
    # Download with progress
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing
    $ProgressPreference = 'Continue'
    
    Write-Host "  Downloaded to: $zipPath" -ForegroundColor Green
    
    # Extract
    Write-Host "Extracting..." -ForegroundColor Yellow
    $tempExtract = Join-Path $ffmpegDir "temp"
    Expand-Archive -Path $zipPath -DestinationPath $tempExtract -Force
    
    # Find the binaries (they're usually in a subfolder)
    $binFolder = Get-ChildItem -Path $tempExtract -Directory | Select-Object -First 1
    $binPath = Join-Path $binFolder.FullName "bin"
    
    # Copy executables to main ffmpeg directory
    Copy-Item -Path "$binPath\ffmpeg.exe" -Destination $ffmpegDir -Force
    Copy-Item -Path "$binPath\ffprobe.exe" -Destination $ffmpegDir -Force
    Copy-Item -Path "$binPath\ffplay.exe" -Destination $ffmpegDir -Force -ErrorAction SilentlyContinue
    
    # Cleanup
    Remove-Item -Path $tempExtract -Recurse -Force
    Remove-Item -Path $zipPath -Force
    
    Write-Host "  Extracted to: $ffmpegDir" -ForegroundColor Green
    
    # Add to PATH for current session
    $env:Path = "$ffmpegDir;$env:Path"
    Write-Host ""
    Write-Host "FFmpeg installed successfully!" -ForegroundColor Green
    
    # Verify installation
    $version = & "$ffmpegExe" -version 2>&1 | Select-String "ffmpeg version" | Select-Object -First 1
    Write-Host "Version: $version" -ForegroundColor Gray
    
    Write-Host ""
    Write-Host "Added to current session PATH." -ForegroundColor Green
    Write-Host ""
    Write-Host "To make permanent, run:" -ForegroundColor Yellow
    Write-Host '  [Environment]::SetEnvironmentVariable("Path", "$env:Path", [System.EnvironmentVariableTarget]::User)' -ForegroundColor Gray
    
} catch {
    Write-Host "ERROR: Failed to download/install FFmpeg" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    Write-Host ""
    Write-Host "Manual installation:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://ffmpeg.org/download.html" -ForegroundColor Gray
    Write-Host "2. Extract ffmpeg.exe and ffprobe.exe to: $ffmpegDir" -ForegroundColor Gray
    Write-Host "3. Add $ffmpegDir to your PATH environment variable" -ForegroundColor Gray
    
    exit 1
}
