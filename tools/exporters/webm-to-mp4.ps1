# WebM to MP4 Converter
# For social media compatibility

param(
    [Parameter(Mandatory=$true)]
    [string]$In,
    
    [Parameter(Mandatory=$false)]
    [string]$Out
)

# Generate output filename if not provided
if (-not $Out) { 
    $Out = $In -replace '\.webm$','.mp4' 
}

# Check if input file exists
if (-not (Test-Path $In)) {
    Write-Host "Error: Input file not found: $In" -ForegroundColor Red
    exit 1
}

# Check if ffmpeg is available
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Error: FFmpeg not found. Install from: https://ffmpeg.org/download.html" -ForegroundColor Red
    exit 1
}

Write-Host "Converting WebM to MP4..." -ForegroundColor Yellow
Write-Host "Input:  $In" -ForegroundColor Cyan
Write-Host "Output: $Out" -ForegroundColor Cyan

# Convert with social media optimized settings
ffmpeg -y -i "$In" `
    -c:v libx264 `
    -preset slow `
    -crf 22 `
    -pix_fmt yuv420p `
    -movflags +faststart `
    -c:a aac `
    -b:a 128k `
    "$Out"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Conversion complete!" -ForegroundColor Green
    
    # Show file sizes
    $inSize = [math]::Round((Get-Item $In).Length / 1MB, 2)
    $outSize = [math]::Round((Get-Item $Out).Length / 1MB, 2)
    
    Write-Host "Input size:  ${inSize}MB" -ForegroundColor Gray
    Write-Host "Output size: ${outSize}MB" -ForegroundColor Gray
} else {
    Write-Host "✗ Conversion failed" -ForegroundColor Red
}