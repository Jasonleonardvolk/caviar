# Encode-WowPack-Direct.ps1
# Direct encoding without parameter passing issues

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     WOW Pack Direct Encoder                   " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# Hard-coded paths
$inputDir = "D:\Dev\kha\content\wowpack\input"
$outputDir = "D:\Dev\kha\tori_ui_svelte\static\media\wow"

# Create output directory
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

# Video 1: holo_flux_loop
$input1 = "$inputDir\holo_flux_loop.mov"
if (Test-Path $input1) {
    Write-Host "[1/3] Encoding holo_flux_loop..." -ForegroundColor Cyan
    
    # HEVC
    Write-Host "  - HEVC HDR10..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input1`" -c:v libx265 -preset medium -crf 18 -pix_fmt yuv420p10le -tag:v hvc1 -an `"$outputDir\holo_flux_loop_hdr10.mp4`" 2>&1"
    
    # AV1
    Write-Host "  - AV1 10-bit..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input1`" -c:v libsvtav1 -preset 8 -crf 35 -pix_fmt yuv420p10le -an `"$outputDir\holo_flux_loop_av1.mp4`" 2>&1"
    
    # SDR
    Write-Host "  - H.264 SDR..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input1`" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -an `"$outputDir\holo_flux_loop_sdr.mp4`" 2>&1"
    
    Write-Host "  [OK] holo_flux_loop complete" -ForegroundColor Green
}

# Video 2: mach_lightfield
$input2 = "$inputDir\mach_lightfield.mov"
if (Test-Path $input2) {
    Write-Host "[2/3] Encoding mach_lightfield..." -ForegroundColor Cyan
    
    # HEVC
    Write-Host "  - HEVC HDR10..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input2`" -c:v libx265 -preset medium -crf 18 -pix_fmt yuv420p10le -tag:v hvc1 -an `"$outputDir\mach_lightfield_hdr10.mp4`" 2>&1"
    
    # AV1
    Write-Host "  - AV1 10-bit..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input2`" -c:v libsvtav1 -preset 8 -crf 35 -pix_fmt yuv420p10le -an `"$outputDir\mach_lightfield_av1.mp4`" 2>&1"
    
    # SDR
    Write-Host "  - H.264 SDR..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input2`" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -an `"$outputDir\mach_lightfield_sdr.mp4`" 2>&1"
    
    Write-Host "  [OK] mach_lightfield complete" -ForegroundColor Green
}

# Video 3: kinetic_logo_parade
$input3 = "$inputDir\kinetic_logo_parade.mov"
if (Test-Path $input3) {
    Write-Host "[3/3] Encoding kinetic_logo_parade..." -ForegroundColor Cyan
    
    # HEVC
    Write-Host "  - HEVC HDR10..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input3`" -c:v libx265 -preset medium -crf 18 -pix_fmt yuv420p10le -tag:v hvc1 -an `"$outputDir\kinetic_logo_parade_hdr10.mp4`" 2>&1"
    
    # AV1
    Write-Host "  - AV1 10-bit..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input3`" -c:v libsvtav1 -preset 8 -crf 35 -pix_fmt yuv420p10le -an `"$outputDir\kinetic_logo_parade_av1.mp4`" 2>&1"
    
    # SDR
    Write-Host "  - H.264 SDR..." -ForegroundColor Yellow
    cmd /c "ffmpeg -y -i `"$input3`" -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -an `"$outputDir\kinetic_logo_parade_sdr.mp4`" 2>&1"
    
    Write-Host "  [OK] kinetic_logo_parade complete" -ForegroundColor Green
}

# Create manifest
Write-Host ""
Write-Host "Creating manifest..." -ForegroundColor Yellow

$manifest = @{
    version = 1
    updated = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ss")
    clips = @(
        @{
            id = "holo_flux_loop"
            label = "Holo Flux Loop"
            sources = @(
                @{ type = "hevc"; url = "/media/wow/holo_flux_loop_hdr10.mp4"; codecs = "hvc1.2.4.L120.B0" }
                @{ type = "av1"; url = "/media/wow/holo_flux_loop_av1.mp4"; codecs = "av01.0.08M.10" }
                @{ type = "h264"; url = "/media/wow/holo_flux_loop_sdr.mp4"; codecs = "avc1.640028" }
            )
        },
        @{
            id = "mach_lightfield"
            label = "Mach Lightfield"
            sources = @(
                @{ type = "hevc"; url = "/media/wow/mach_lightfield_hdr10.mp4"; codecs = "hvc1.2.4.L120.B0" }
                @{ type = "av1"; url = "/media/wow/mach_lightfield_av1.mp4"; codecs = "av01.0.08M.10" }
                @{ type = "h264"; url = "/media/wow/mach_lightfield_sdr.mp4"; codecs = "avc1.640028" }
            )
        },
        @{
            id = "kinetic_logo_parade"
            label = "Kinetic Logo Parade"
            sources = @(
                @{ type = "hevc"; url = "/media/wow/kinetic_logo_parade_hdr10.mp4"; codecs = "hvc1.2.4.L120.B0" }
                @{ type = "av1"; url = "/media/wow/kinetic_logo_parade_av1.mp4"; codecs = "av01.0.08M.10" }
                @{ type = "h264"; url = "/media/wow/kinetic_logo_parade_sdr.mp4"; codecs = "avc1.640028" }
            )
        }
    )
}

$manifestPath = "$outputDir\wow.manifest.json"
$manifest | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $manifestPath

Write-Host "Manifest created: $manifestPath" -ForegroundColor Green

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Encoding Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Output files in: $outputDir" -ForegroundColor Yellow
Write-Host ""
Write-Host "Test at:" -ForegroundColor Cyan
Write-Host "  http://localhost:3000/hologram?clip=holo_flux_loop" -ForegroundColor White
Write-Host "  http://localhost:3000/hologram?clip=mach_lightfield" -ForegroundColor White
Write-Host "  http://localhost:3000/hologram?clip=kinetic_logo_parade" -ForegroundColor White
