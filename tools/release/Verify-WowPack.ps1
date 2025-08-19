# Verify-WowPack.ps1
# Verifies all WOW Pack encoded assets are present with correct HDR atoms

param([string]$StaticWow = "D:\Dev\kha\tori_ui_svelte\static\media\wow")

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     WOW Pack Verification                     " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check manifest exists
$manifestPath = "$StaticWow\wow.manifest.json"
if(-not (Test-Path $manifestPath)) {
    Write-Host "[ERROR] Manifest not found: $manifestPath" -ForegroundColor Red
    Write-Host "Run encoding first!" -ForegroundColor Yellow
    exit 1
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

Write-Host "Manifest version: $($manifest.version)" -ForegroundColor Gray
Write-Host "Last updated:     $($manifest.updated)" -ForegroundColor Gray
Write-Host "Total clips:      $($manifest.clips.Count)" -ForegroundColor Gray
Write-Host ""

if($manifest.clips.Count -eq 0) {
    Write-Host "[WARNING] No clips in manifest. Encode your videos first!" -ForegroundColor Yellow
    exit
}

Write-Host "Verifying encoded files and HDR atoms..." -ForegroundColor Yellow
Write-Host ""

# Table header
"{0,-25} {1,-8} {2,-12} {3,-10} {4,-8} {5,-15} {6}" -f "Clip", "Format", "Codec", "Pixel Fmt", "Size MB", "Transfer", "HDR"
"{0,-25} {1,-8} {2,-12} {3,-10} {4,-8} {5,-15} {6}" -f ("-"*25), ("-"*8), ("-"*12), ("-"*10), ("-"*8), ("-"*15), ("-"*3)

$totalFiles = 0
$missingFiles = 0
$hdrCount = 0
$atomIssues = @()

foreach($clip in $manifest.clips) {
    foreach($source in $clip.sources) {
        $totalFiles++
        $fileName = Split-Path $source.url -Leaf
        $filePath = Join-Path $StaticWow $fileName
        
        if(Test-Path $filePath) {
            $fileInfo = Get-Item $filePath
            $sizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
            
            # Deep probe for HDR metadata and atoms
            $probeOutput = & ffprobe -v quiet -print_format json -show_streams -show_format "$filePath" 2>$null | ConvertFrom-Json
            $videoStream = $probeOutput.streams | Where-Object {$_.codec_type -eq 'video'} | Select-Object -First 1
            
            $transfer = $videoStream.color_transfer
            $primaries = $videoStream.color_primaries
            $pixFmt = $videoStream.pix_fmt
            $codecName = $videoStream.codec_name
            $brand = $probeOutput.format.major_brand
            
            # Check for mastering display metadata (HDR10 side data)
            $hasMasteringData = $false
            if($videoStream.side_data_list) {
                foreach($sideData in $videoStream.side_data_list) {
                    if($sideData.side_data_type -match "Mastering") {
                        $hasMasteringData = $true
                        break
                    }
                }
            }
            
            # Validate based on type
            $isValid = $true
            $hdrStatus = "No"
            
            switch($source.type) {
                "hevc" {
                    # HEVC should have HDR10 atoms
                    if($transfer -eq "smpte2084" -and $primaries -eq "bt2020" -and $hasMasteringData) {
                        $hdrStatus = "HDR10"
                        $hdrCount++
                    } else {
                        $atomIssues += "$fileName missing HDR10 atoms (transfer=$transfer, primaries=$primaries, mastering=$hasMasteringData)"
                        $isValid = $false
                    }
                }
                "av1" {
                    # AV1 should have 10-bit and HDR signaling
                    if($pixFmt -eq "yuv420p10le" -and $transfer -eq "smpte2084") {
                        $hdrStatus = "HDR"
                        $hdrCount++
                    } else {
                        $atomIssues += "$fileName missing HDR signaling (pix_fmt=$pixFmt, transfer=$transfer)"
                        $isValid = $false
                    }
                }
                "h264" {
                    # SDR should be bt709 with no HDR metadata
                    if($transfer -eq "bt709" -and -not $hasMasteringData) {
                        $hdrStatus = "SDR"
                    } else {
                        $atomIssues += "$fileName should be SDR but has (transfer=$transfer, mastering=$hasMasteringData)"
                        $isValid = $false
                    }
                }
            }
            
            $statusColor = if($isValid) {"Green"} else {"Yellow"}
            Write-Host ("{0,-25} {1,-8} {2,-12} {3,-10} {4,-8} {5,-15} {6}" -f $clip.id, $source.type.ToUpper(), $codecName, $pixFmt, $sizeMB, $transfer, $hdrStatus) -ForegroundColor $statusColor
        } else {
            "{0,-25} {1,-8} {2,-12} {3,-10} {4,-8} {5,-15} {6}" -f $clip.id, $source.type.ToUpper(), "MISSING!", "-", "-", "-", "-"
            Write-Host "  [ERROR] Missing file: $fileName" -ForegroundColor Red
            $missingFiles++
        }
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Verification Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Total files expected:  $totalFiles" -ForegroundColor Gray
Write-Host "Files present:         $($totalFiles - $missingFiles)" -ForegroundColor $(if($missingFiles -eq 0) {"Green"} else {"Yellow"})
Write-Host "Files missing:         $missingFiles" -ForegroundColor $(if($missingFiles -eq 0) {"Green"} else {"Red"})
Write-Host "HDR variants:          $hdrCount" -ForegroundColor Gray

if($atomIssues.Count -gt 0) {
    Write-Host ""
    Write-Host "Atom/Metadata Issues:" -ForegroundColor Yellow
    foreach($issue in $atomIssues) {
        Write-Host "  - $issue" -ForegroundColor Yellow
    }
}

# Check HLS if present
$hlsCount = 0
foreach($clip in $manifest.clips) {
    if($clip.hls) {
        $hlsPath = "D:\Dev\kha\tori_ui_svelte\static\media\hls\$($clip.id)"
        if(Test-Path "$hlsPath\playlist.m3u8") {
            $hlsCount++
            $segments = (Get-ChildItem "$hlsPath\*.m4s" -ErrorAction SilentlyContinue).Count
            Write-Host "HLS: $($clip.id) - $segments segments" -ForegroundColor Gray
        }
    }
}

if($hlsCount -gt 0) {
    Write-Host ""
    Write-Host "HLS streams generated: $hlsCount" -ForegroundColor Green
}

Write-Host ""

# Validate manifest structure
$manifestValid = $true
foreach($clip in $manifest.clips) {
    $hasAv1 = $false
    $hasHevc = $false
    $hasSdr = $false
    
    foreach($source in $clip.sources) {
        if($source.type -eq "av1" -and $source.codecs -eq "av01.0.08M.10") { $hasAv1 = $true }
        if($source.type -eq "hevc" -and $source.codecs -eq "hvc1.2.4.L120.B0") { $hasHevc = $true }
        if($source.type -eq "h264" -and $source.codecs -eq "avc1.640028") { $hasSdr = $true }
    }
    
    if(-not ($hasAv1 -and $hasHevc -and $hasSdr)) {
        Write-Host "[WARNING] $($clip.id) missing variants (AV1=$hasAv1, HEVC=$hasHevc, SDR=$hasSdr)" -ForegroundColor Yellow
        $manifestValid = $false
    }
}

if($missingFiles -eq 0 -and $atomIssues.Count -eq 0 -and $manifestValid) {
    Write-Host "[OK] All files verified with correct HDR atoms!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Ready to test at:" -ForegroundColor Cyan
    foreach($clip in $manifest.clips) {
        Write-Host "  http://localhost:3000/hologram?clip=$($clip.id)" -ForegroundColor White
    }
} else {
    Write-Host "[WARNING] Some issues detected. Review details above." -ForegroundColor Yellow
}

Write-Host ""
