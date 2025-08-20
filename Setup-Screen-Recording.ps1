# Enable and Configure Windows Game Bar for Screen Recording
Write-Host "🎮 Setting up Windows Game Bar for screen recording..." -ForegroundColor Cyan

# Enable Game Bar
Write-Host "Enabling Game Bar..." -ForegroundColor Yellow
Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\GameDVR" -Name "AppCaptureEnabled" -Value 1 -Type DWord -Force
Set-ItemProperty -Path "HKCU:\System\GameConfigStore" -Name "GameDVR_Enabled" -Value 1 -Type DWord -Force

# Set recording quality
Write-Host "Configuring recording settings..." -ForegroundColor Yellow
Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\GameDVR" -Name "VideoEncodingBitrateMode" -Value 1 -Type DWord -Force
Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\GameDVR" -Name "VideoEncodingResolutionMode" -Value 1 -Type DWord -Force
Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\GameDVR" -Name "AudioEncodingBitrate" -Value 128000 -Type DWord -Force

Write-Host "✅ Game Bar configured!" -ForegroundColor Green
Write-Host ""
Write-Host "📹 HOW TO USE WINDOWS GAME BAR:" -ForegroundColor Magenta
Write-Host "1. Press Win+G to open Game Bar"
Write-Host "2. Click the camera icon or press Win+Alt+R to start/stop recording"
Write-Host "3. Recordings save to: Videos\Captures folder"
Write-Host ""
Write-Host "🎯 QUICK RECORDING TIPS:" -ForegroundColor Yellow
Write-Host "- Record just the browser window for cleaner videos"
Write-Host "- Use Win+Alt+M to toggle microphone if narrating"
Write-Host "- Recordings are saved as MP4 - no conversion needed!"
Write-Host ""

# Open Videos\Captures folder
$capturesPath = [Environment]::GetFolderPath("MyVideos") + "\Captures"
if (-not (Test-Path $capturesPath)) {
    New-Item -ItemType Directory -Path $capturesPath -Force | Out-Null
}

$openFolder = Read-Host "Open Captures folder now? (y/n)"
if ($openFolder -eq 'y') {
    Start-Process explorer $capturesPath
}

Write-Host ""
Write-Host "Alternative: OBS Studio (if you want more control)" -ForegroundColor Cyan
Write-Host "Download from: https://obsproject.com/" -ForegroundColor White
