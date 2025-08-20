param([string]$ProjectRoot = "D:\Dev\kha")

Write-Host "`n===============================================" -ForegroundColor Magenta
Write-Host "    WOW PACK ULTIMATE PRODUCTION CHECK" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Magenta

function Ok($m){Write-Host "[OK] $m" -f Green}
function Info($m){Write-Host "[INFO] $m" -f Cyan}
function Warn($m){Write-Host "[WARN] $m" -f Yellow}
function Fail($m){Write-Host "[FAIL] $m" -f Red}
function Header($m){Write-Host "`n$m" -f Magenta; Write-Host ("=" * $m.Length) -f Magenta}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$allGood = $true

# Create detailed report
$reportData = @{
    timestamp = $timestamp
    checks = @{}
    summary = @{}
}

Header "1. FFMPEG INSTALLATION CHECK"
$ffmpegPath = "D:\Dev\kha\tools\ffmpeg\ffmpeg.exe"
if (Test-Path $ffmpegPath) {
    Ok "FFmpeg binary found at tools\ffmpeg\"
    
    # Add to PATH for this session if not already there
    if ($env:Path -notlike "*D:\Dev\kha\tools\ffmpeg*") {
        $env:Path = "D:\Dev\kha\tools\ffmpeg;" + $env:Path
        Info "Added FFmpeg to PATH for this session"
    }
    
    # Get detailed version info
    try {
        $versionFull = & $ffmpegPath -version 2>&1 
        $version = $versionFull | Select-String "ffmpeg version" | Select-Object -First 1
        $config = $versionFull | Select-String "configuration:" | Select-Object -First 1
        
        if ($version) {
            Info "Version: $version"
            $reportData.checks.ffmpeg = @{status="OK"; version="$version"}
        }
        
        # Check for ProRes support
        $codecs = & $ffmpegPath -codecs 2>&1 | Select-String "prores"
        if ($codecs) {
            Ok "ProRes codec support verified"
        }
    } catch {
        Warn "Could not get FFmpeg details"
    }
} else {
    Fail "FFmpeg not found - run .\tools\encode\Install-FFmpeg.ps1"
    $allGood = $false
    $reportData.checks.ffmpeg = @{status="MISSING"}
}

Header "2. PRORES MASTERS VALIDATION"
$inputDir = "D:\Dev\kha\content\wowpack\input"
$requiredFiles = @(
    @{name="holo_flux_loop.mov"; minSize=100MB},
    @{name="mach_lightfield.mov"; minSize=100MB},
    @{name="kinetic_logo_parade.mov"; minSize=50MB}
)

$masterCount = 0
$totalMasterSize = 0
$masters = @()

foreach ($fileInfo in $requiredFiles) {
    $filePath = Join-Path $inputDir $fileInfo.name
    if (Test-Path $filePath) {
        $item = Get-Item $filePath
        $size = $item.Length
        $totalMasterSize += $size
        
        if ($size -ge $fileInfo.minSize) {
            $sizeGB = [math]::Round($size/1GB, 2)
            Ok "$($fileInfo.name) - $sizeGB GB"
            $masterCount++
            
            # Get video properties using ffprobe
            if (Test-Path "D:\Dev\kha\tools\ffmpeg\ffprobe.exe") {
                $probeCmd = "& 'D:\Dev\kha\tools\ffmpeg\ffprobe.exe' -v error -show_entries stream=codec_name,width,height,r_frame_rate,pix_fmt -of json '$filePath' 2>&1"
                try {
                    $probeResult = Invoke-Expression $probeCmd | ConvertFrom-Json
                    if ($probeResult.streams) {
                        $stream = $probeResult.streams[0]
                        Info "  Format: $($stream.codec_name), $($stream.width)x$($stream.height), $($stream.pix_fmt)"
                    }
                } catch {}
            }
            
            $masters += @{
                name = $fileInfo.name
                size = $size
                status = "OK"
            }
        } else {
            $sizeMB = [math]::Round($size/1MB, 2)
            Warn "$($fileInfo.name) exists but seems small ($sizeMB MB)"
        }
    } else {
        Fail "$($fileInfo.name) missing"
        $allGood = $false
    }
}

$totalSizeGB = [math]::Round($totalMasterSize/1GB, 2)
Info "`nMasters: $masterCount/3 present"
Info "Total size: $totalSizeGB GB"
$reportData.checks.masters = @{count=$masterCount; total=3; files=$masters}

# Check for MP4 companions
$mp4Files = Get-ChildItem "$inputDir\*.mp4" -ErrorAction SilentlyContinue
if ($mp4Files) {
    Ok "Bonus: $($mp4Files.Count) MP4 preview versions available"
    foreach ($mp4 in $mp4Files) {
        $sizeMB = [math]::Round($mp4.Length/1MB, 1)
        Info "  * $($mp4.Name) ($sizeMB MB)"
    }
}

Header "3. ENCODED OUTPUT VALIDATION"

# Check AV1 outputs
$av1Dir = "D:\Dev\kha\content\wowpack\video\av1"
$av1Count = 0
if (Test-Path $av1Dir) {
    $av1Files = Get-ChildItem "$av1Dir\*.mp4" -ErrorAction SilentlyContinue
    if ($av1Files) {
        Ok "AV1 Encodes: $($av1Files.Count) files"
        foreach ($av1 in $av1Files) {
            $sizeMB = [math]::Round($av1.Length/1MB, 1)
            Info "  * $($av1.Name) ($sizeMB MB)"
            $av1Count++
        }
    }
} else {
    Warn "AV1 directory not found"
}

# Check HDR10 outputs
$hdr10Dir = "D:\Dev\kha\content\wowpack\video\hdr10"
$hdrCount = 0
$sdrCount = 0
if (Test-Path $hdr10Dir) {
    $hdrFiles = Get-ChildItem "$hdr10Dir\*_hdr10.mp4" -ErrorAction SilentlyContinue
    $sdrFiles = Get-ChildItem "$hdr10Dir\*_sdr.mp4" -ErrorAction SilentlyContinue
    
    if ($hdrFiles) {
        Ok "HDR10 Encodes: $($hdrFiles.Count) files"
        foreach ($hdr in $hdrFiles) {
            $sizeMB = [math]::Round($hdr.Length/1MB, 1)
            Info "  * $($hdr.Name) ($sizeMB MB)"
            $hdrCount++
        }
    }
    
    if ($sdrFiles) {
        Ok "SDR Encodes: $($sdrFiles.Count) files"
        foreach ($sdr in $sdrFiles) {
            $sizeMB = [math]::Round($sdr.Length/1MB, 1)
            Info "  * $($sdr.Name) ($sizeMB MB)"
            $sdrCount++
        }
    }
} else {
    Warn "HDR10 directory not found"
}

$reportData.checks.outputs = @{
    av1 = $av1Count
    hdr10 = $hdrCount
    sdr = $sdrCount
}

Header "4. PIPELINE SCRIPTS CHECK"
$encodeScripts = @(
    @{name="Check-ProRes-Masters.ps1"; desc="ProRes validation"},
    @{name="Batch-Encode-Simple.ps1"; desc="Basic encoding"},
    @{name="Build-WowPack.ps1"; desc="Full production"},
    @{name="Build-SocialPack.ps1"; desc="Social media pack"},
    @{name="Encode-WowPack-Direct.ps1"; desc="Direct encoding"}
)

$scriptCount = 0
foreach ($script in $encodeScripts) {
    $scriptPath = "D:\Dev\kha\tools\encode\$($script.name)"
    if (Test-Path $scriptPath) {
        Ok "$($script.name) - $($script.desc)"
        $scriptCount++
    } else {
        Fail "$($script.name) missing"
    }
}

Info "`nScripts: $scriptCount/$($encodeScripts.Count) available"

Header "5. ADDITIONAL RESOURCES"

# Check grading LUTs
$gradingDir = "D:\Dev\kha\content\wowpack\grading"
if (Test-Path $gradingDir) {
    $luts = Get-ChildItem "$gradingDir\*.cube" -ErrorAction SilentlyContinue
    if ($luts) {
        Ok "Color Grading: $($luts.Count) LUTs available"
    }
}

# Check stills
$stillsDir = "D:\Dev\kha\content\wowpack\stills"
if (Test-Path $stillsDir) {
    $stills = Get-ChildItem "$stillsDir\*" -File -ErrorAction SilentlyContinue
    if ($stills) {
        Ok "Still Frames: $($stills.Count) images extracted"
    }
}

# Check social pack
$socialDir = "D:\Dev\kha\content\socialpack"
if (Test-Path $socialDir) {
    $socialContent = Get-ChildItem "$socialDir\*" -Recurse -File -ErrorAction SilentlyContinue
    if ($socialContent) {
        Ok "Social Pack: $($socialContent.Count) assets ready"
    }
}

Header "6. QUICK SYSTEM TEST"

# Test encode capability
$testOutput = "D:\Dev\kha\content\wowpack\test_encode_$(Get-Date -Format 'yyyyMMdd_HHmmss').mp4"
try {
    Write-Host "  Running quick encode test..." -NoNewline
    $testCmd = "& '$ffmpegPath' -y -f lavfi -t 1 -i 'testsrc2=s=1920x1080:r=60' -c:v libx264 -preset ultrafast -pix_fmt yuv420p '$testOutput' 2>&1"
    $result = Invoke-Expression $testCmd
    
    if (Test-Path $testOutput) {
        $testSize = (Get-Item $testOutput).Length
        $testSizeKB = [math]::Round($testSize/1KB, 2)
        Ok " Success! ($testSizeKB KB test file)"
        Remove-Item $testOutput -Force
    } else {
        Warn " Encode test failed"
    }
} catch {
    Warn " Could not test encoding: $_"
}

# Generate comprehensive report
$reportData.summary = @{
    ffmpeg_ready = (Test-Path $ffmpegPath)
    masters_ready = ($masterCount -eq 3)
    outputs_ready = (($av1Count -gt 0) -or ($hdrCount -gt 0))
    scripts_ready = ($scriptCount -eq $encodeScripts.Count)
    total_master_size_gb = $totalSizeGB
    encoded_files = @{
        av1 = $av1Count
        hdr10 = $hdrCount
        sdr = $sdrCount
    }
}

# Save report
$reportDir = Join-Path $ProjectRoot "verification_reports"
if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}
$reportPath = Join-Path $reportDir "wowpack_status_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$reportData | ConvertTo-Json -Depth 5 | Set-Content $reportPath
Info "`nDetailed report saved: $reportPath"

# Final summary
Write-Host "`n===============================================" -ForegroundColor Magenta
if ($masterCount -eq 3 -and $scriptCount -eq $encodeScripts.Count) {
    if (($av1Count -gt 0) -or ($hdrCount -gt 0)) {
        Write-Host "WOW PACK FULLY OPERATIONAL & ENCODED!" -ForegroundColor Green
        Write-Host "`nALL SYSTEMS GO FOR PRODUCTION!" -ForegroundColor Green
        Info "`nYour WOW Pack includes:"
        Info "  * $masterCount ProRes masters ($totalSizeGB GB)"
        Info "  * $av1Count AV1 encodes (next-gen codec)"
        Info "  * $hdrCount HDR10 + $sdrCount SDR encodes"
        Info "  * Ready for demo at /hologram"
    } else {
        Write-Host "WOW PACK READY - ENCODING NEEDED" -ForegroundColor Yellow
        Write-Host "`nTo generate final outputs:" -ForegroundColor Yellow
        Write-Host "  cd D:\Dev\kha\tools\encode" -ForegroundColor White
        Write-Host "  .\Batch-Encode-Simple.ps1" -ForegroundColor White
    }
} elseif ($masterCount -gt 0) {
    Warn "WOW Pack partially ready ($masterCount/3 masters)"
    if ($masterCount -lt 3) {
        Info "`nTo create missing masters, run:"
        Info "  cd D:\Dev\kha\tools\encode"
        Info "  .\Generate-Test-ProRes.ps1"
    }
} else {
    Fail "WOW Pack not ready - setup required"
    Info "`nQuick setup:"
    Info "  1. cd D:\Dev\kha\tools\encode"
    Info "  2. .\Install-FFmpeg.ps1"
    Info "  3. .\Generate-Test-ProRes.ps1"
}

Write-Host "===============================================" -ForegroundColor Magenta
Write-Host ""

# Exit with appropriate code
if ($allGood -and $masterCount -eq 3) {
    exit 0
} else {
    exit 1
}