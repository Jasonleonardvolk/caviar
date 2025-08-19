# Simple ffmpeg installer
$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$downloadPath = "$env:TEMP\ffmpeg.zip"

Write-Host "Downloading ffmpeg..." -ForegroundColor Cyan
Invoke-WebRequest -Uri $ffmpegUrl -OutFile $downloadPath -UseBasicParsing

Write-Host "Extracting..." -ForegroundColor Cyan
Expand-Archive -Path $downloadPath -DestinationPath "C:\" -Force

# Find and rename the folder
$ffmpegFolder = Get-ChildItem -Path "C:\" -Directory | Where-Object { $_.Name -like "ffmpeg-*" } | Select-Object -First 1

if ($ffmpegFolder) {
    if (Test-Path "C:\ffmpeg") {
        Remove-Item -Path "C:\ffmpeg" -Recurse -Force
    }
    Move-Item -Path $ffmpegFolder.FullName -Destination "C:\ffmpeg" -Force
    Write-Host "ffmpeg installed to C:\ffmpeg" -ForegroundColor Green
}

# Cleanup
Remove-Item -Path $downloadPath -Force -ErrorAction SilentlyContinue

# Test
if (Test-Path "C:\ffmpeg\bin\ffmpeg.exe") {
    Write-Host "Success! ffmpeg is installed." -ForegroundColor Green
    & "C:\ffmpeg\bin\ffmpeg.exe" -version | Select-Object -First 1
} else {
    Write-Host "Error: ffmpeg.exe not found" -ForegroundColor Red
}
