# Batch-Encode-Simple.ps1
# Encodes all ProRes .mov masters to HEVC HDR10, AV1, and H.264 SDR

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     WOW Pack ProRes to MP4 Encoder            " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Get absolute paths
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$toolsDir = Split-Path -Parent $scriptDir
$repoDir = Split-Path -Parent $toolsDir
$inputDir = Join-Path $repoDir "content\wowpack\input"

# Check for ProRes .mov masters
$videos = @(
    "holo_flux_loop",
    "mach_lightfield",
    "kinetic_logo_parade"
)

Write-Host "Checking for ProRes .mov masters..." -ForegroundColor Yellow
Write-Host "  Input directory: $inputDir" -ForegroundColor Gray
Write-Host ""

$foundAll = $true
$videoFiles = @{}

foreach($video in $videos) {
    $movPath = Join-Path $inputDir "$video.mov"
    
    if(Test-Path $movPath) {
        Write-Host "  [OK] Found: $video.mov" -ForegroundColor Green
        
        # Quick probe to verify it's ProRes
        $probeOutput = & ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,pix_fmt -of default=noprint_wrappers=1 $movPath 2>$null
        if ($probeOutput -match "prores") {
            Write-Host "       Verified: ProRes codec" -ForegroundColor Gray
        } else {
            Write-Host "       Warning: Not ProRes, but will attempt encoding" -ForegroundColor Yellow
        }
        $videoFiles[$video] = $movPath
    } else {
        Write-Host "  [X] Missing: $video.mov" -ForegroundColor Red
        $foundAll = $false
    }
}

if(-not $foundAll) {
    Write-Host ""
    Write-Host "ERROR: ProRes .mov masters missing!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please add your ProRes masters to:" -ForegroundColor Yellow
    Write-Host "  $inputDir" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Expected files:" -ForegroundColor Yellow
    Write-Host "  - holo_flux_loop.mov (ProRes 422/4444)" -ForegroundColor Gray
    Write-Host "  - mach_lightfield.mov (ProRes 422/4444)" -ForegroundColor Gray
    Write-Host "  - kinetic_logo_parade.mov (ProRes 422/4444)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

Write-Host ""
Write-Host "All ProRes masters found!" -ForegroundColor Green
Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Starting HDR encoding pipeline..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

$successCount = 0
$failedVideos = @()

# Change to script directory for Build-WowPack.ps1
Set-Location $scriptDir

# Encode video 1 - holo_flux_loop with HLS
if ($videoFiles.ContainsKey("holo_flux_loop")) {
    Write-Host "[1/3] Encoding holo_flux_loop..." -ForegroundColor Cyan
    Write-Host "      Options: HEVC HDR10 + AV1 + SDR + HLS" -ForegroundColor Gray
    Write-Host "      Input: $($videoFiles["holo_flux_loop"])" -ForegroundColor Gray
    try {
        & .\Build-WowPack.ps1 -Basename "holo_flux_loop" -InputFile $videoFiles["holo_flux_loop"] -Framerate 60 -DoSDR -MakeHLS
        Write-Host "  [OK] holo_flux_loop complete!" -ForegroundColor Green
        $successCount++
    } catch {
        Write-Host "  [FAIL] holo_flux_loop failed: $_" -ForegroundColor Red
        $failedVideos += "holo_flux_loop"
    }
    Write-Host ""
}

# Encode video 2 - mach_lightfield
if ($videoFiles.ContainsKey("mach_lightfield")) {
    Write-Host "[2/3] Encoding mach_lightfield..." -ForegroundColor Cyan
    Write-Host "      Options: HEVC HDR10 + AV1 + SDR" -ForegroundColor Gray
    Write-Host "      Input: $($videoFiles["mach_lightfield"])" -ForegroundColor Gray
    try {
        & .\Build-WowPack.ps1 -Basename "mach_lightfield" -InputFile $videoFiles["mach_lightfield"] -Framerate 60 -DoSDR
        Write-Host "  [OK] mach_lightfield complete!" -ForegroundColor Green
        $successCount++
    } catch {
        Write-Host "  [FAIL] mach_lightfield failed: $_" -ForegroundColor Red
        $failedVideos += "mach_lightfield"
    }
    Write-Host ""
}

# Encode video 3 - kinetic_logo_parade
if ($videoFiles.ContainsKey("kinetic_logo_parade")) {
    Write-Host "[3/3] Encoding kinetic_logo_parade..." -ForegroundColor Cyan
    Write-Host "      Options: HEVC HDR10 + AV1 + SDR" -ForegroundColor Gray
    Write-Host "      Input: $($videoFiles["kinetic_logo_parade"])" -ForegroundColor Gray
    try {
        & .\Build-WowPack.ps1 -Basename "kinetic_logo_parade" -InputFile $videoFiles["kinetic_logo_parade"] -Framerate 60 -DoSDR
        Write-Host "  [OK] kinetic_logo_parade complete!" -ForegroundColor Green
        $successCount++
    } catch {
        Write-Host "  [FAIL] kinetic_logo_parade failed: $_" -ForegroundColor Red
        $failedVideos += "kinetic_logo_parade"
    }
    Write-Host ""
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Encoding Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Successfully encoded: $successCount / 3" -ForegroundColor $(if($successCount -eq 3) {"Green"} else {"Yellow"})

if($failedVideos.Count -gt 0) {
    Write-Host ""
    Write-Host "Failed videos:" -ForegroundColor Red
    foreach($failed in $failedVideos) {
        Write-Host "  - $failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Running verification..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Run verification from release directory
$releaseDir = Join-Path $toolsDir "release"
Set-Location $releaseDir
& .\Verify-WowPack.ps1

if($successCount -eq 3) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host "*** All videos encoded successfully! ***" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Test your HDR clips at:" -ForegroundColor Cyan
    Write-Host "  http://localhost:3000/hologram?clip=holo_flux_loop" -ForegroundColor White
    Write-Host "  http://localhost:3000/hologram?clip=mach_lightfield" -ForegroundColor White
    Write-Host "  http://localhost:3000/hologram?clip=kinetic_logo_parade" -ForegroundColor White
    Write-Host ""
    Write-Host "Codec selection:" -ForegroundColor Yellow
    Write-Host "  - Hardware AV1 10-bit support: AV1" -ForegroundColor Gray
    Write-Host "  - Hardware HEVC HDR10 support: HEVC" -ForegroundColor Gray
    Write-Host "  - Fallback: H.264 SDR" -ForegroundColor Gray
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
