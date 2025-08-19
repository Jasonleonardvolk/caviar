# Generate-Test-ProRes.ps1
# Creates ProRes test masters from existing MP4 files for pipeline validation
# Locked to ProRes 422 HQ 10-bit 60fps for consistent testing

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Generating ProRes Test Masters from MP4s     " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check for FFmpeg first
$ffmpegPath = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpegPath) {
    # Try local install
    $localFFmpeg = "D:\Dev\kha\tools\ffmpeg\ffmpeg.exe"
    if (Test-Path $localFFmpeg) {
        $env:Path = "D:\Dev\kha\tools\ffmpeg;$env:Path"
        Write-Host "Using local FFmpeg installation" -ForegroundColor Green
    } else {
        Write-Host "ERROR: FFmpeg not found!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please run:" -ForegroundColor Yellow
        Write-Host "  .\Install-FFmpeg.ps1" -ForegroundColor White
        Write-Host ""
        Write-Host "Or install FFmpeg manually and add to PATH" -ForegroundColor Gray
        exit 1
    }
}

$inputDir = "D:\Dev\kha\content\wowpack\input"
Set-Location $inputDir

# Check for existing MP4s
$mp4Files = @(
    @{mp4="holo_flux_loop.mp4"; mov="holo_flux_loop.mov"},
    @{mp4="mach_lightfield.mp4"; mov="mach_lightfield.mov"},
    @{mp4="kinetic_logo_parade.mp4"; mov="kinetic_logo_parade.mov"}
)

$converted = 0

foreach($file in $mp4Files) {
    if(Test-Path $file.mp4) {
        Write-Host "Converting: $($file.mp4) -> $($file.mov)" -ForegroundColor Yellow
        Write-Host "  Target: ProRes 422 HQ, 10-bit, 60fps" -ForegroundColor Gray
        
        # Build FFmpeg arguments as array
        $ffmpegArgs = @(
            "-y",
            "-i", $file.mp4,
            "-c:v", "prores_ks",
            "-profile:v", "3",
            "-pix_fmt", "yuv422p10le",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-r", "60",
            "-c:a", "pcm_s16le",
            $file.mov
        )
        
        # Execute FFmpeg
        & ffmpeg $ffmpegArgs
        
        if($LASTEXITCODE -eq 0) {
            # Verify the output
            $probeJson = & ffprobe -v quiet -print_format json -show_streams $file.mov 2>$null | ConvertFrom-Json
            $videoStream = $probeJson.streams | Where-Object {$_.codec_type -eq 'video'} | Select-Object -First 1
            
            Write-Host "  [OK] Created $($file.mov)" -ForegroundColor Green
            if ($videoStream) {
                Write-Host "       Codec: $($videoStream.codec_name)" -ForegroundColor Gray
                Write-Host "       Pixel Format: $($videoStream.pix_fmt)" -ForegroundColor Gray
                Write-Host "       Frame Rate: $($videoStream.r_frame_rate)" -ForegroundColor Gray
            }
            $converted++
        } else {
            Write-Host "  [FAIL] Could not create $($file.mov)" -ForegroundColor Red
        }
    } else {
        Write-Host "  [SKIP] $($file.mp4) not found" -ForegroundColor Gray
    }
    Write-Host ""
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Converted: $converted / 3 files" -ForegroundColor $(if($converted -eq 3) {"Green"} else {"Yellow"})

if($converted -eq 3) {
    Write-Host ""
    Write-Host "ProRes test masters ready!" -ForegroundColor Green
    Write-Host "All files: ProRes 422 HQ, 10-bit, 60fps, Rec.709" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  .\Check-ProRes-Masters.ps1" -ForegroundColor White
    Write-Host "  .\Batch-Encode-Simple.ps1" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "Not all files converted. Check for missing MP4s." -ForegroundColor Yellow
}
