# Check-WowPack-Status.ps1
# Quick status check for WOW Pack setup

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "      WOW Pack Status Check            " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check input directory
$inputDir = "..\..\content\wowpack\input"
$expectedVideos = @(
    "holo_flux_loop",
    "mach_lightfield", 
    "kinetic_logo_parade"
)

Write-Host "INPUT VIDEOS:" -ForegroundColor Yellow
Write-Host "Location: D:\Dev\kha\content\wowpack\input\" -ForegroundColor Gray
Write-Host ""

$foundCount = 0
foreach($video in $expectedVideos) {
    $movPath = "$inputDir\$video.mov"
    $mp4Path = "$inputDir\$video.mp4"
    
    if(Test-Path $movPath) {
        Write-Host "  [OK] $video.mov" -ForegroundColor Green
        $foundCount++
    } elseif(Test-Path $mp4Path) {
        Write-Host "  [OK] $video.mp4" -ForegroundColor Green
        $foundCount++
    } else {
        Write-Host "  [ ] $video (.mov or .mp4) - NOT FOUND" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Status: $foundCount of 3 videos found" -ForegroundColor $(if($foundCount -eq 3) {"Green"} elseif($foundCount -gt 0) {"Yellow"} else {"Red"})

# Check encoded outputs
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ENCODED OUTPUTS:" -ForegroundColor Yellow

$outputDir = "..\..\tori_ui_svelte\static\media\wow"
if(Test-Path "$outputDir\wow.manifest.json") {
    $manifest = Get-Content "$outputDir\wow.manifest.json" -Raw | ConvertFrom-Json
    $clipCount = $manifest.clips.Count
    
    if($clipCount -gt 0) {
        Write-Host "Manifest has $clipCount encoded clip(s):" -ForegroundColor Green
        foreach($clip in $manifest.clips) {
            Write-Host "  - $($clip.id) ($($clip.sources.Count) formats)" -ForegroundColor Gray
        }
    } else {
        Write-Host "No clips encoded yet (manifest is empty)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Manifest not found - no videos encoded yet" -ForegroundColor Red
}

# Check FFmpeg
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "SYSTEM REQUIREMENTS:" -ForegroundColor Yellow

$ffmpegOK = Get-Command ffmpeg -ErrorAction SilentlyContinue
$ffprobeOK = Get-Command ffprobe -ErrorAction SilentlyContinue

if($ffmpegOK) {
    Write-Host "  [OK] FFmpeg installed" -ForegroundColor Green
} else {
    Write-Host "  [X] FFmpeg not found - install with: winget install ffmpeg" -ForegroundColor Red
}

if($ffprobeOK) {
    Write-Host "  [OK] FFprobe installed" -ForegroundColor Green
} else {
    Write-Host "  [X] FFprobe not found" -ForegroundColor Red
}

# Next steps
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Yellow

if($foundCount -eq 0) {
    Write-Host ""
    Write-Host "1. Place your video files in:" -ForegroundColor White
    Write-Host "   D:\Dev\kha\content\wowpack\input\" -ForegroundColor Gray
    Write-Host ""
    Write-Host "   Expected files:" -ForegroundColor White
    Write-Host "   - holo_flux_loop.mov (or .mp4)" -ForegroundColor Gray
    Write-Host "   - mach_lightfield.mov (or .mp4)" -ForegroundColor Gray
    Write-Host "   - kinetic_logo_parade.mov (or .mp4)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Run: .\Batch-Encode-All.ps1" -ForegroundColor White
} elseif($foundCount -lt 3) {
    Write-Host ""
    Write-Host "1. Add missing video files to input directory" -ForegroundColor White
    Write-Host "2. Run: .\Batch-Encode-All.ps1" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "All videos ready! Run:" -ForegroundColor Green
    Write-Host "  .\Batch-Encode-All.ps1" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
