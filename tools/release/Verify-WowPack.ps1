param([string]$ProjectRoot = "D:\Dev\kha")

Write-Host "`n🎬 WOW PACK PRODUCTION STATUS" -ForegroundColor Cyan
Write-Host "===============================" -ForegroundColor Cyan

function Ok($m){Write-Host "[OK] $m" -f Green}
function Info($m){Write-Host "[i] $m" -f Cyan}
function Warn($m){Write-Host "[!] $m" -f Yellow}
function Fail($m){Write-Host "[X] $m" -f Red}

# 1. Check FFmpeg
$ffmpegPath = "D:\Dev\kha\tools\ffmpeg\ffmpeg.exe"
if (Test-Path $ffmpegPath) {
    Ok "FFmpeg installed at tools\ffmpeg\"
    # Add to PATH for this session if not already there
    if ($env:Path -notlike "*D:\Dev\kha\tools\ffmpeg*") {
        $env:Path = "D:\Dev\kha\tools\ffmpeg;$env:Path"
        Info "  Added FFmpeg to PATH for this session"
    }
    
    # Test FFmpeg works
    $version = & $ffmpegPath -version 2>&1 | Select-String "ffmpeg version" | Select-Object -First 1
    if ($version) {
        Info "  Version: $version"
    }
} else {
    Fail "FFmpeg not found - run .\tools\encode\Install-FFmpeg.ps1"
    exit 1
}

# 2. Check ProRes Masters
Write-Host "`n📹 ProRes Masters Status:" -ForegroundColor Cyan
$inputDir = "D:\Dev\kha\content\wowpack\input"
$requiredFiles = @(
    "holo_flux_loop.mov",
    "mach_lightfield.mov", 
    "kinetic_logo_parade.mov"
)

$masterCount = 0
foreach ($file in $requiredFiles) {
    $filePath = Join-Path $inputDir $file
    if (Test-Path $filePath) {
        $size = (Get-Item $filePath).Length / 1MB
        Ok "  ✓ $file ($('{0:N1}' -f $size) MB)"
        $masterCount++
    } else {
        Fail "  ✗ $file missing"
    }
}

Info "`n  Masters: $masterCount/3 present"

# 3. Check MP4 versions (bonus)
$mp4Files = Get-ChildItem "$inputDir\*.mp4" -ErrorAction SilentlyContinue
if ($mp4Files) {
    Info "  Bonus: $($mp4Files.Count) MP4 versions also available"
}

# 4. Check output directories
Write-Host "`n📁 Output Structure:" -ForegroundColor Cyan
$outputDirs = @{
    "Video Output" = "D:\Dev\kha\content\wowpack\video"
    "Grading LUTs" = "D:\Dev\kha\content\wowpack\grading"
    "Stills" = "D:\Dev\kha\content\wowpack\stills"
    "Social Pack" = "D:\Dev\kha\content\socialpack"
}

foreach ($dir in $outputDirs.GetEnumerator()) {
    if (Test-Path $dir.Value) {
        $files = (Get-ChildItem $dir.Value -File -ErrorAction SilentlyContinue).Count
        Ok "  $($dir.Key): $files files"
    } else {
        Warn "  $($dir.Key): Not created yet"
    }
}

# 5. Check encoding scripts
Write-Host "`n🔧 Encoding Scripts:" -ForegroundColor Cyan
$encodeScripts = @(
    "Check-ProRes-Masters.ps1",
    "Batch-Encode-Simple.ps1",
    "Build-WowPack.ps1",
    "Build-SocialPack.ps1"
)

$scriptCount = 0
foreach ($script in $encodeScripts) {
    $scriptPath = "D:\Dev\kha\tools\encode\$script"
    if (Test-Path $scriptPath) {
        Ok "  ✓ $script"
        $scriptCount++
    } else {
        Fail "  ✗ $script"
    }
}

# 6. Quick encode test
Write-Host "`n🚀 Quick Encode Test:" -ForegroundColor Cyan
$testOutput = "D:\Dev\kha\content\wowpack\test_encode.mp4"
try {
    # Quick 1-second test encode
    $testCmd = "& '$ffmpegPath' -y -f lavfi -t 1 -i testsrc2=s=640x360 -c:v libx264 -preset ultrafast '$testOutput' 2>&1"
    $result = Invoke-Expression $testCmd
    if (Test-Path $testOutput) {
        Ok "  FFmpeg encode test passed"
        Remove-Item $testOutput -Force
    } else {
        Warn "  FFmpeg test encode failed"
    }
} catch {
    Warn "  Could not test FFmpeg encoding"
}

# Summary
Write-Host "`n===============================" -ForegroundColor Cyan
if ($masterCount -eq 3 -and $scriptCount -eq 4) {
    Write-Host "✅ WOW PACK READY FOR PRODUCTION!" -ForegroundColor Green
    Write-Host "`nNext steps to generate outputs:" -ForegroundColor Yellow
    Write-Host "  cd D:\Dev\kha\tools\encode" -ForegroundColor White
    Write-Host "  .\Batch-Encode-Simple.ps1     # Basic H264/HEVC/AV1" -ForegroundColor White
    Write-Host "  .\Build-WowPack.ps1           # Full production pack" -ForegroundColor White
    Write-Host "  .\Build-SocialPack.ps1        # Social media optimized" -ForegroundColor White
} elseif ($masterCount -gt 0) {
    Warn "⚠️  WOW Pack partially ready ($masterCount/3 masters)"
} else {
    Fail "❌ WOW Pack not ready - masters missing"
}

Write-Host ""