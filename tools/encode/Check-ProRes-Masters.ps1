# Check-ProRes-Masters.ps1
# Strictly verifies ProRes .mov masters are ready for HDR encoding
# FAILS if input is not ProRes 10-bit

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     ProRes Masters Pre-Flight Check           " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$inputDir = "D:\Dev\kha\content\wowpack\input"
$expectedVideos = @(
    "holo_flux_loop.mov",
    "mach_lightfield.mov", 
    "kinetic_logo_parade.mov"
)

Write-Host "Checking for ProRes masters in:" -ForegroundColor Yellow
Write-Host "  $inputDir" -ForegroundColor Gray
Write-Host ""

$foundCount = 0
$readyCount = 0
$failureReasons = @()

foreach($video in $expectedVideos) {
    $videoPath = Join-Path $inputDir $video
    $basename = [System.IO.Path]::GetFileNameWithoutExtension($video)
    
    if(Test-Path $videoPath) {
        $foundCount++
        Write-Host "[FOUND] $video" -ForegroundColor Green
        
        # Run detailed probe
        $probeJson = & ffprobe -v quiet -print_format json -show_streams "$videoPath" 2>$null | ConvertFrom-Json
        
        if($probeJson -and $probeJson.streams) {
            $videoStream = $probeJson.streams | Where-Object {$_.codec_type -eq 'video'} | Select-Object -First 1
            
            if($videoStream) {
                $isProRes = $videoStream.codec_name -match "prores"
                $is10Bit = $videoStream.pix_fmt -match "10"
                $validPixFmt = $videoStream.pix_fmt -in @("yuv422p10le", "yuv444p10le")
                
                Write-Host "  Codec:       $($videoStream.codec_name)" -ForegroundColor $(if($isProRes) {"Green"} else {"Red"})
                Write-Host "  Resolution:  $($videoStream.width)x$($videoStream.height)" -ForegroundColor Gray
                Write-Host "  Pixel Fmt:   $($videoStream.pix_fmt)" -ForegroundColor $(if($validPixFmt) {"Green"} else {"Red"})
                
                # Check color space
                if($videoStream.color_primaries) {
                    $isRec709 = $videoStream.color_primaries -eq "bt709"
                    Write-Host "  Color Space: $($videoStream.color_primaries)" -ForegroundColor $(if($isRec709) {"Green"} else {"Yellow"})
                } else {
                    Write-Host "  Color Space: [not specified, assuming bt709]" -ForegroundColor Yellow
                }
                
                # STRICT ENFORCEMENT
                if(-not $isProRes) {
                    Write-Host "  Status:      FAILED - Not ProRes!" -ForegroundColor Red
                    $failureReasons += "$basename is not ProRes (found: $($videoStream.codec_name))"
                } elseif(-not $validPixFmt) {
                    Write-Host "  Status:      FAILED - Not 10-bit ProRes!" -ForegroundColor Red
                    $failureReasons += "$basename is not 10-bit ProRes (found: $($videoStream.pix_fmt))"
                } else {
                    Write-Host "  Status:      READY - ProRes 10-bit confirmed!" -ForegroundColor Green
                    $readyCount++
                }
                
                # File size
                $fileInfo = Get-Item $videoPath
                $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
                Write-Host "  File Size:   $sizeMB MB" -ForegroundColor Gray
            }
        }
        Write-Host ""
    } else {
        Write-Host "[MISSING] $video" -ForegroundColor Red
        $failureReasons += "$video not found"
        Write-Host ""
    }
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Files found:     $foundCount / 3" -ForegroundColor $(if($foundCount -eq 3) {"Green"} else {"Red"})
Write-Host "Ready to encode: $readyCount / 3" -ForegroundColor $(if($readyCount -eq 3) {"Green"} else {"Red"})
Write-Host ""

if($failureReasons.Count -gt 0) {
    Write-Host "FAILURES DETECTED:" -ForegroundColor Red
    foreach($reason in $failureReasons) {
        Write-Host "  - $reason" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "ABORTING: Masters must be ProRes 10-bit .mov files!" -ForegroundColor Red
    Write-Host ""
    Write-Host "ACTION REQUIRED:" -ForegroundColor Yellow
    Write-Host "1. Export your videos from After Effects/Premiere as:" -ForegroundColor Yellow
    Write-Host "   - Codec: ProRes 422 HQ or ProRes 4444" -ForegroundColor Gray
    Write-Host "   - Color: Rec.709, 10-bit" -ForegroundColor Gray
    Write-Host "   - Resolution: 1920x1080 or higher" -ForegroundColor Gray
    Write-Host "   - Frame Rate: Any (will be converted to 60fps)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Name them exactly:" -ForegroundColor Yellow
    Write-Host "   - holo_flux_loop.mov" -ForegroundColor Gray
    Write-Host "   - mach_lightfield.mov" -ForegroundColor Gray
    Write-Host "   - kinetic_logo_parade.mov" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Place them in:" -ForegroundColor Yellow
    Write-Host "   $inputDir" -ForegroundColor Gray
    
    exit 1  # FAIL THE SCRIPT
} elseif($readyCount -eq 3) {
    Write-Host "All masters validated - ProRes 10-bit confirmed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step:" -ForegroundColor Cyan
    Write-Host "  .\Batch-Encode-Simple.ps1" -ForegroundColor White
    exit 0
} else {
    Write-Host "Some files need attention. Check the details above." -ForegroundColor Yellow
    exit 1
}
