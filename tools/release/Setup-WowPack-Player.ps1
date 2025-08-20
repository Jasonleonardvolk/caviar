param([string]$ProjectRoot = "D:\Dev\kha")

Write-Host "`n🎬 SETTING UP WOW PACK PLAYER FOR /HOLOGRAM" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

function Ok($m){Write-Host "[✅] $m" -f Green}
function Info($m){Write-Host "[ℹ️] $m" -f Cyan}
function Warn($m){Write-Host "[⚠️] $m" -f Yellow}
function Fail($m){Write-Host "[❌] $m" -f Red}

# 1. Ensure output directory exists
$outputDir = "D:\Dev\kha\content\wowpack\output"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Ok "Created output directory"
} else {
    Ok "Output directory exists"
}

# 2. Copy encoded files to output directory
Write-Host "`n📦 Copying encoded videos to output directory..." -ForegroundColor Yellow

$copiedCount = 0

# Copy H264/MP4 versions from input (web-playable versions)
$inputMp4 = Get-ChildItem "D:\Dev\kha\content\wowpack\input\*.mp4" -ErrorAction SilentlyContinue
if ($inputMp4) {
    foreach ($file in $inputMp4) {
        Copy-Item $file.FullName $outputDir -Force
        Info "  Copied: $($file.Name)"
        $copiedCount++
    }
}

# Copy AV1 versions (next-gen codec)
$av1Files = Get-ChildItem "D:\Dev\kha\content\wowpack\video\av1\*.mp4" -ErrorAction SilentlyContinue
if ($av1Files) {
    foreach ($file in $av1Files) {
        $newName = $file.BaseName -replace "_av1", "_av1"
        Copy-Item $file.FullName (Join-Path $outputDir "$newName.mp4") -Force
        Info "  Copied: $newName.mp4 (AV1)"
        $copiedCount++
    }
}

# Copy HDR10 versions (premium quality)
$hdr10Files = Get-ChildItem "D:\Dev\kha\content\wowpack\video\hdr10\*_hdr10.mp4" -ErrorAction SilentlyContinue
if ($hdr10Files) {
    foreach ($file in $hdr10Files) {
        Copy-Item $file.FullName $outputDir -Force
        Info "  Copied: $($file.Name) (HDR10)"
        $copiedCount++
    }
}

# Copy SDR versions (compatibility)
$sdrFiles = Get-ChildItem "D:\Dev\kha\content\wowpack\video\hdr10\*_sdr.mp4" -ErrorAction SilentlyContinue
if ($sdrFiles) {
    foreach ($file in $sdrFiles) {
        Copy-Item $file.FullName $outputDir -Force
        Info "  Copied: $($file.Name) (SDR)"
        $copiedCount++
    }
}

if ($copiedCount -gt 0) {
    Ok "Copied $copiedCount video files to output directory"
} else {
    Warn "No encoded videos found to copy"
    Info "Tip: Run .\tools\encode\Batch-Encode-Simple.ps1 to generate encodes"
}

# 3. Show what's available
Write-Host "`n📹 Available videos in output directory:" -ForegroundColor Yellow
$outputFiles = Get-ChildItem "$outputDir\*.mp4", "$outputDir\*.webm", "$outputDir\*.mov" -ErrorAction SilentlyContinue
if ($outputFiles) {
    foreach ($file in $outputFiles) {
        $sizeMB = [math]::Round($file.Length / 1MB, 1)
        Info "  • $($file.Name) ($sizeMB MB)"
    }
    Ok "Total: $($outputFiles.Count) video files ready"
} else {
    Warn "No video files in output directory yet"
}

# 4. Create a quick test file if none exist
if ($outputFiles.Count -eq 0) {
    Write-Host "`n🎨 Creating a test video for immediate demo..." -ForegroundColor Yellow
    
    $ffmpegPath = "D:\Dev\kha\tools\ffmpeg\ffmpeg.exe"
    if (Test-Path $ffmpegPath) {
        $testFile = Join-Path $outputDir "holo_flux_loop_demo.mp4"
        
        # Create a simple test pattern video
        $cmd = "& '$ffmpegPath' -y -f lavfi -r 30 -t 5 " +
               "-i `"testsrc2=s=1920x1080:r=30`" " +
               "-vf `"eq=brightness=0.1:saturation=1.5`" " +
               "-c:v libx264 -preset fast -pix_fmt yuv420p " +
               "`"$testFile`" 2>&1"
        
        Write-Host "  Generating demo video..." -NoNewline
        $result = Invoke-Expression $cmd
        
        if (Test-Path $testFile) {
            Ok " Created demo video!"
            Info "  $testFile"
        } else {
            Fail " Could not create demo video"
        }
    } else {
        Warn "FFmpeg not found - cannot create demo video"
    }
}

# 5. Verify API endpoints
Write-Host "`n🔌 Verifying API endpoints..." -ForegroundColor Yellow

$apiFiles = @(
    "D:\Dev\kha\frontend\src\routes\api\wowpack\list\+server.ts",
    "D:\Dev\kha\frontend\src\routes\api\wowpack\file\[name]\+server.ts"
)

$apiReady = $true
foreach ($apiFile in $apiFiles) {
    if (Test-Path $apiFile) {
        Ok "  $(Split-Path -Leaf (Split-Path -Parent $apiFile))"
    } else {
        Fail "  Missing: $apiFile"
        $apiReady = $false
    }
}

# 6. Verify component integration
$componentFile = "D:\Dev\kha\frontend\src\lib\components\WowpackPlayer.svelte"
if (Test-Path $componentFile) {
    Ok "WowpackPlayer component ready"
} else {
    Fail "WowpackPlayer component missing"
}

# Check if integrated into hologram page
$hologramPage = "D:\Dev\kha\frontend\src\routes\hologram\+page.svelte"
$integrated = $false
if (Test-Path $hologramPage) {
    $content = Get-Content $hologramPage -Raw
    if ($content -match "WowpackPlayer") {
        Ok "Component integrated into /hologram page"
        $integrated = $true
    } else {
        Warn "Component not integrated into /hologram page"
    }
}

# 7. Final instructions
Write-Host "`n============================================" -ForegroundColor Cyan
if ($copiedCount -gt 0 -and $apiReady -and $integrated) {
    Write-Host "✨ WOW PACK PLAYER READY FOR DEMO! ✨" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Yellow
    Write-Host "  1. Restart dev server (if running):" -ForegroundColor White
    Write-Host "     cd D:\Dev\kha\frontend" -ForegroundColor Gray
    Write-Host "     pnpm dev" -ForegroundColor Gray
    Write-Host "`n  2. Open http://localhost:5173/hologram" -ForegroundColor White
    Write-Host "`n  3. Click the video tabs to play demos!" -ForegroundColor White
    
    Write-Host "`n🎬 Demo talking points:" -ForegroundColor Cyan
    Write-Host "  • Professional ProRes masters (5.56 GB)" -ForegroundColor White
    Write-Host "  • Multiple codec variants (H.264, AV1, HDR10)" -ForegroundColor White
    Write-Host "  • Streaming API with range support" -ForegroundColor White
    Write-Host "  • Adaptive player with codec detection" -ForegroundColor White
} else {
    Write-Host "⚠️ WOW Pack Player needs attention" -ForegroundColor Yellow
    
    if ($copiedCount -eq 0) {
        Write-Host "`nTo generate videos:" -ForegroundColor Yellow
        Write-Host "  cd D:\Dev\kha\tools\encode" -ForegroundColor White
        Write-Host "  .\Batch-Encode-Simple.ps1" -ForegroundColor White
    }
    
    if (-not $apiReady) {
        Write-Host "`nAPI endpoints need setup - check file creation" -ForegroundColor Yellow
    }
    
    if (-not $integrated) {
        Write-Host "`nComponent needs integration into hologram page" -ForegroundColor Yellow
    }
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""