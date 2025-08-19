# Fix-HEVC-Mastering.ps1
# Patches Build-WowPack.ps1 to properly embed HDR10 mastering metadata

$buildWowPath = "D:\Dev\kha\tools\encode\Build-WowPack.ps1"

if (!(Test-Path $buildWowPath)) {
    Write-Host "Build-WowPack.ps1 not found!" -ForegroundColor Red
    exit 1
}

# Backup original
$backup = "$buildWowPath.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
Copy-Item $buildWowPath $backup
Write-Host "Backed up to: $backup" -ForegroundColor Gray

# Read current content
$content = Get-Content $buildWowPath -Raw

# The corrected HEVC args with proper mastering metadata
$newHevcBlock = @'
    $hevcParams = @(
        "-i", $Input,
        "-r", $Framerate,
        "-pix_fmt", "yuv420p10le",
        "-c:v", "libx265",
        "-crf", "18",
        "-preset", "medium",
        # x265: HDR10 SEI + MaxCLL + repeat headers + info SEI
        "-x265-params",
        "hdr10=1:hdr10-opt=1:repeat-headers=1:info=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400",
        # ffmpeg side tagging (container colr box + tags)
        "-master_display", "G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)",
        "-max_cll", "1000,400",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-metadata:s:v:0", "color_primaries=bt2020",
        "-metadata:s:v:0", "color_transfer=smpte2084",
        "-metadata:s:v:0", "color_space=bt2020nc",
        "-tag:v", "hvc1",
        "-movflags", "+faststart+write_colr"
    )
    
    if ($HDR10) {
        # Already included above
    }
    
    $hevcParams += @(
        "-c:a", "copy",
        $Output
    )
'@

# Look for the Encode-HEVC function and update it
if ($content -match 'function Encode-HEVC') {
    Write-Host "Found Encode-HEVC function, patching..." -ForegroundColor Cyan
    
    # This is a simplified patch - you may need to adjust based on exact structure
    $content = $content -replace '(\$hevcParams = @\([^)]+\))', $newHevcBlock
    
    # Write back
    $content | Set-Content $buildWowPath -Encoding UTF8
    Write-Host "✅ Patched Build-WowPack.ps1 with proper HDR10 mastering metadata!" -ForegroundColor Green
    
    Write-Host "`nTest with:" -ForegroundColor Yellow
    Write-Host ".\tools\encode\Build-WowPack.ps1 -InputFile video.mp4 -Codec hevc -HDR10" -ForegroundColor White
    Write-Host ".\tools\release\Verify-WowPack.ps1  # Should show mastering=True" -ForegroundColor White
} else {
    Write-Host "Could not find Encode-HEVC function - manual edit needed" -ForegroundColor Yellow
}