# WebM to MP4 converter for social media
param(
    [Parameter(Mandatory=$true)]
    [string]$In,
    
    [string]$Out = ""
)

if (-not $Out) {
    $Out = $In -replace '\.webm$','.mp4'
}

Write-Host "Converting $In to $Out..." -ForegroundColor Cyan

# Check if ffmpeg exists
$ffmpegPath = (Get-Command ffmpeg -ErrorAction SilentlyContinue).Path
if (-not $ffmpegPath) {
    Write-Host "ERROR: ffmpeg not found. Please install ffmpeg first." -ForegroundColor Red
    Write-Host "Download from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
    exit 1
}

# Convert with optimized settings for social media
ffmpeg -y -i "$In" `
    -c:v libx264 `
    -preset fast `
    -crf 23 `
    -pix_fmt yuv420p `
    -movflags +faststart `
    -c:a aac `
    -b:a 128k `
    "$Out"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Successfully converted to: $Out" -ForegroundColor Green
    
    # Get file info
    $fileInfo = Get-Item $Out
    $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
    Write-Host "File size: ${sizeMB}MB" -ForegroundColor Cyan
} else {
    Write-Host "❌ Conversion failed" -ForegroundColor Red
}
