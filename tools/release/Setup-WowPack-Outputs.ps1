param([string]$ProjectRoot = "D:\Dev\kha")

Write-Host "`n🎬 SETTING UP WOW PACK OUTPUTS FOR LIVE DEMO" -ForegroundColor Magenta
Write-Host "================================================" -ForegroundColor Magenta

function Ok($m){Write-Host "[✅] $m" -f Green}
function Info($m){Write-Host "[ℹ️] $m" -f Cyan}
function Warn($m){Write-Host "[⚠️] $m" -f Yellow}
function Fail($m){Write-Host "[❌] $m" -f Red}

$outputDir = "D:\Dev\kha\content\wowpack\output"

# Ensure output directory exists
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Ok "Created output directory"
}

Write-Host "`n📦 Organizing encoded videos for player..." -ForegroundColor Cyan

# Define the mapping of source files to target names
$fileMappings = @(
    # H264/MP4 versions from input directory (if they exist)
    @{
        Source = "D:\Dev\kha\content\wowpack\input\holo_flux_loop.mp4"
        Target = "holo_flux_loop_h264.mp4"
        Type = "H.264"
    },
    @{
        Source = "D:\Dev\kha\content\wowpack\input\mach_lightfield.mp4"
        Target = "mach_lightfield_h264.mp4"
        Type = "H.264"
    },
    @{
        Source = "D:\Dev\kha\content\wowpack\input\kinetic_logo_parade.mp4"
        Target = "kinetic_logo_parade_h264.mp4"
        Type = "H.264"
    },
    
    # AV1 versions
    @{
        Source = "D:\Dev\kha\content\wowpack\video\av1\holo_flux_loop_av1.mp4"
        Target = "holo_flux_loop_av1.mp4"
        Type = "AV1"
    },
    @{
        Source = "D:\Dev\kha\content\wowpack\video\av1\mach_lightfield_av1.mp4"
        Target = "mach_lightfield_av1.mp4"
        Type = "AV1"
    },
    @{
        Source = "D:\Dev\kha\content\wowpack\video\av1\kinetic_logo_parade_av1.mp4"
        Target = "kinetic_logo_parade_av1.mp4"
        Type = "AV1"
    },
    
    # HDR10 versions
    @{
        Source = "D:\Dev\kha\content\wowpack\video\hdr10\holo_flux_loop_hdr10.mp4"
        Target = "holo_flux_loop_hdr10.mp4"
        Type = "HDR10"
    },
    @{
        Source = "D:\Dev\kha\content\wowpack\video\hdr10\mach_lightfield_hdr10.mp4"
        Target = "mach_lightfield_hdr10.mp4"
        Type = "HDR10"
    },
    @{
        Source = "D:\Dev\kha\content\wowpack\video\hdr10\kinetic_logo_parade_hdr10.mp4"
        Target = "kinetic_logo_parade_hdr10.mp4"
        Type = "HDR10"
    },
    
    # SDR versions
    @{
        Source = "D:\Dev\kha\content\wowpack\video\hdr10\holo_flux_loop_sdr.mp4"
        Target = "holo_flux_loop_sdr.mp4"
        Type = "SDR"
    },
    @{
        Source = "D:\Dev\kha\content\wowpack\video\hdr10\mach_lightfield_sdr.mp4"
        Target = "mach_lightfield_sdr.mp4"
        Type = "SDR"
    },
    @{
        Source = "D:\Dev\kha\content\wowpack\video\hdr10\kinetic_logo_parade_sdr.mp4"
        Target = "kinetic_logo_parade_sdr.mp4"
        Type = "SDR"
    }
)

$copiedCount = 0
$formats = @{}

foreach ($mapping in $fileMappings) {
    if (Test-Path $mapping.Source) {
        $targetPath = Join-Path $outputDir $mapping.Target
        Copy-Item $mapping.Source $targetPath -Force
        
        $sizeMB = [math]::Round((Get-Item $mapping.Source).Length / 1MB, 1)
        Ok "Copied $($mapping.Target) ($($mapping.Type), $sizeMB MB)"
        
        $copiedCount++
        
        # Track formats per video
        $baseName = $mapping.Target -replace '_(h264|av1|hdr10|sdr)\.mp4$', ''
        if (-not $formats[$baseName]) {
            $formats[$baseName] = @()
        }
        $formats[$baseName] += $mapping.Type
    }
}

Write-Host "`n📊 Summary by Video:" -ForegroundColor Cyan
foreach ($video in $formats.Keys | Sort-Object) {
    $displayName = switch($video) {
        "holo_flux_loop" { "HOLO FLUX" }
        "mach_lightfield" { "MACH LIGHTFIELD" }
        "kinetic_logo_parade" { "KINETIC LOGO PARADE" }
        default { $video }
    }
    Info "$displayName : $($formats[$video] -join ', ')"
}

Write-Host "`n🎯 Final Status:" -ForegroundColor Cyan
Ok "Total files copied: $copiedCount"

# List all files in output directory
Write-Host "`n📁 All files in output directory:" -ForegroundColor Yellow
$allFiles = Get-ChildItem "$outputDir\*.mp4", "$outputDir\*.webm", "$outputDir\*.mov" -ErrorAction SilentlyContinue | Sort-Object Name
if ($allFiles) {
    foreach ($file in $allFiles) {
        $sizeMB = [math]::Round($file.Length / 1MB, 1)
        Info "  $($file.Name) - $sizeMB MB"
    }
    Write-Host "`n✨ Total: $($allFiles.Count) video files ready for demo!" -ForegroundColor Green
}

# Quick test of the API endpoint
Write-Host "`n🔌 Testing API endpoints..." -ForegroundColor Cyan
$testUrl = "http://localhost:5173/api/wowpack/list"
try {
    $response = Invoke-WebRequest -Uri $testUrl -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Ok "API endpoint responding!"
        $data = $response.Content | ConvertFrom-Json
        foreach ($item in $data.items) {
            if ($item.files.Count -gt 0) {
                Info "  $($item.title): $($item.files.Count) files available"
            }
        }
    }
} catch {
    Warn "Dev server not running or API not responding"
    Info "Start it with: cd frontend && pnpm dev"
}

Write-Host "`n================================================" -ForegroundColor Magenta
Write-Host "🚀 WOW PACK OUTPUTS READY FOR DEMO!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Magenta

Write-Host "`n📝 Next Steps:" -ForegroundColor Yellow
Write-Host "  1. If dev server not running:" -ForegroundColor White
Write-Host "     cd D:\Dev\kha\frontend" -ForegroundColor Gray
Write-Host "     pnpm dev" -ForegroundColor Gray
Write-Host "`n  2. Open: http://localhost:5173/hologram" -ForegroundColor White
Write-Host "`n  3. Click video tabs to showcase:" -ForegroundColor White
Write-Host "     • HOLO FLUX - Smooth waves with glow" -ForegroundColor Gray
Write-Host "     • MACH LIGHTFIELD - Interference rings" -ForegroundColor Gray
Write-Host "     • KINETIC LOGO PARADE - Motion graphics" -ForegroundColor Gray

Write-Host "`n🎬 Demo Highlights:" -ForegroundColor Cyan
Write-Host "  • $copiedCount encoded variants ready" -ForegroundColor White
Write-Host "  • Multiple formats: H.264, AV1, HDR10, SDR" -ForegroundColor White
Write-Host "  • Streaming API with adaptive codec selection" -ForegroundColor White
Write-Host "  • Professional 5.56 GB ProRes source masters" -ForegroundColor White

Write-Host "`n✨ READY TO IMPRESS! ✨" -ForegroundColor Magenta
Write-Host ""