# Download and install ffmpeg automatically
$ffmpegUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$downloadPath = "$env:TEMP\ffmpeg.zip"
$extractPath = "C:\"

Write-Host "📥 Downloading ffmpeg..." -ForegroundColor Cyan
try {
    # Download ffmpeg
    Invoke-WebRequest -Uri $ffmpegUrl -OutFile $downloadPath -UseBasicParsing
    Write-Host "✅ Download complete!" -ForegroundColor Green
    
    # Extract the zip file
    Write-Host "📦 Extracting ffmpeg..." -ForegroundColor Cyan
    Expand-Archive -Path $downloadPath -DestinationPath $extractPath -Force
    
    # Find the extracted folder (it has a version number in the name)
    $ffmpegFolder = Get-ChildItem -Path $extractPath -Directory | Where-Object { $_.Name -like "ffmpeg-*" } | Select-Object -First 1
    
    if ($ffmpegFolder) {
        # Rename to simple "ffmpeg" folder
        $targetPath = "C:\ffmpeg"
        if (Test-Path $targetPath) {
            Write-Host "⚠️  Removing old ffmpeg installation..." -ForegroundColor Yellow
            Remove-Item -Path $targetPath -Recurse -Force
        }
        
        Move-Item -Path $ffmpegFolder.FullName -Destination $targetPath -Force
        Write-Host "✅ ffmpeg installed to C:\ffmpeg" -ForegroundColor Green
        
        # Verify installation
        if (Test-Path "C:\ffmpeg\bin\ffmpeg.exe") {
            Write-Host "✅ ffmpeg.exe verified at C:\ffmpeg\bin\ffmpeg.exe" -ForegroundColor Green
            
            # Test ffmpeg
            Write-Host "`n🧪 Testing ffmpeg..." -ForegroundColor Cyan
            & "C:\ffmpeg\bin\ffmpeg.exe" -version | Select-Object -First 1
            
            Write-Host "`n✅ ffmpeg is ready to use!" -ForegroundColor Green
            Write-Host "Since C:\ffmpeg\bin is already in your PATH, you can now use ffmpeg from any directory." -ForegroundColor Yellow
        } else {
            Write-Host "❌ Error: ffmpeg.exe not found in expected location" -ForegroundColor Red
        }
    } else {
        Write-Host "❌ Error: Could not find extracted ffmpeg folder" -ForegroundColor Red
    }
    
    # Clean up
    Remove-Item -Path $downloadPath -Force -ErrorAction SilentlyContinue
    Write-Host "`n🧹 Cleaned up temporary files" -ForegroundColor Gray
    
} catch {
    Write-Host "❌ Error downloading/installing ffmpeg: $_" -ForegroundColor Red
    Write-Host "`nYou can manually download from:" -ForegroundColor Yellow
    Write-Host "https://github.com/BtbN/FFmpeg-Builds/releases" -ForegroundColor Cyan
    Write-Host "Look for 'ffmpeg-master-latest-win64-gpl.zip'" -ForegroundColor Cyan
}
