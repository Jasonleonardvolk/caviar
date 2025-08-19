# Add ffmpeg to PATH
$ffmpegPath = "C:\ffmpeg\bin"

# Get current PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Check if ffmpeg is already in PATH
if ($currentPath -notlike "*$ffmpegPath*") {
    # Add ffmpeg to PATH
    $newPath = $currentPath + ";" + $ffmpegPath
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    
    Write-Host "✅ Added $ffmpegPath to PATH" -ForegroundColor Green
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: You need to restart your PowerShell/Command Prompt for changes to take effect!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or run this command in your current session to use it immediately:" -ForegroundColor Cyan
    Write-Host '$env:Path += ";C:\ffmpeg\bin"' -ForegroundColor White
} else {
    Write-Host "✅ $ffmpegPath is already in PATH" -ForegroundColor Green
}

# Verify ffmpeg installation
if (Test-Path "C:\ffmpeg\bin\ffmpeg.exe") {
    Write-Host "✅ ffmpeg.exe found at C:\ffmpeg\bin" -ForegroundColor Green
} else {
    Write-Host "❌ ffmpeg.exe NOT found at C:\ffmpeg\bin" -ForegroundColor Red
    Write-Host "Please download ffmpeg from https://ffmpeg.org/download.html" -ForegroundColor Yellow
    Write-Host "Extract it so that ffmpeg.exe is located at C:\ffmpeg\bin\ffmpeg.exe" -ForegroundColor Yellow
}
