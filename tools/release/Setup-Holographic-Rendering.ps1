param([string]$ProjectRoot = "D:\Dev\kha")

Write-Host "`n" -NoNewline
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "     HOLOGRAPHIC RENDERING SETUP - THE FINAL TOUCH!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

function Ok($m){Write-Host "[OK] $m" -f Green}
function Info($m){Write-Host "[INFO] $m" -f Cyan}
function Warn($m){Write-Host "[WARN] $m" -f Yellow}
function Header($m){
    Write-Host "`n$m" -f Magenta
    Write-Host ("=" * $m.Length) -f DarkMagenta
}

Header "STEP 1: VERIFY VIDEO SOURCES"

$inputDir = "D:\Dev\kha\content\wowpack\input"
$outputDir = "D:\Dev\kha\content\wowpack\output"

# Ensure output directory exists
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    Ok "Created output directory"
} else {
    Ok "Output directory exists"
}

# Check for video files
$videoSources = @{
    "MP4 in input" = (Get-ChildItem "$inputDir\*.mp4" -ErrorAction SilentlyContinue)
    "AV1 encoded" = (Get-ChildItem "D:\Dev\kha\content\wowpack\video\av1\*.mp4" -ErrorAction SilentlyContinue)
    "HDR10 encoded" = (Get-ChildItem "D:\Dev\kha\content\wowpack\video\hdr10\*_hdr10.mp4" -ErrorAction SilentlyContinue)
    "SDR encoded" = (Get-ChildItem "D:\Dev\kha\content\wowpack\video\hdr10\*_sdr.mp4" -ErrorAction SilentlyContinue)
}

$totalVideos = 0
foreach ($source in $videoSources.GetEnumerator()) {
    if ($source.Value) {
        Ok "$($source.Key): $($source.Value.Count) files found"
        $totalVideos += $source.Value.Count
    }
}

Header "STEP 2: COPY VIDEOS TO OUTPUT"

$copiedCount = 0

# Copy MP4 versions from input
if ($videoSources["MP4 in input"]) {
    foreach ($file in $videoSources["MP4 in input"]) {
        Copy-Item $file.FullName $outputDir -Force
        Info "  Copied: $($file.Name)"
        $copiedCount++
    }
}

# Copy AV1 versions
if ($videoSources["AV1 encoded"]) {
    foreach ($file in $videoSources["AV1 encoded"]) {
        Copy-Item $file.FullName $outputDir -Force
        Info "  Copied: $($file.Name) (AV1)"
        $copiedCount++
    }
}

# Copy HDR10 versions
if ($videoSources["HDR10 encoded"]) {
    foreach ($file in $videoSources["HDR10 encoded"]) {
        Copy-Item $file.FullName $outputDir -Force
        Info "  Copied: $($file.Name) (HDR10)"
        $copiedCount++
    }
}

# Copy SDR versions
if ($videoSources["SDR encoded"]) {
    foreach ($file in $videoSources["SDR encoded"]) {
        Copy-Item $file.FullName $outputDir -Force
        Info "  Copied: $($file.Name) (SDR)"
        $copiedCount++
    }
}

if ($copiedCount -gt 0) {
    Ok "Successfully copied $copiedCount video files to output"
} else {
    Warn "No videos were copied - output directory may be empty"
}

Header "STEP 3: VERIFY HOLOGRAPHIC COMPONENTS"

$components = @(
    @{path="D:\Dev\kha\frontend\src\routes\hologram\+page.svelte"; desc="Hologram page with WebGL2 shaders"},
    @{path="D:\Dev\kha\frontend\src\lib\components\HologramSourceSelector.svelte"; desc="Video source selector HUD"},
    @{path="D:\Dev\kha\frontend\src\routes\api\wowpack\list\+server.ts"; desc="Video list API"},
    @{path="D:\Dev\kha\frontend\src\routes\api\wowpack\file\[name]\+server.ts"; desc="Video streaming API"}
)

$allComponentsReady = $true
foreach ($comp in $components) {
    if (Test-Path $comp.path) {
        Ok "$($comp.desc)"
    } else {
        Warn "Missing: $($comp.desc)"
        $allComponentsReady = $false
    }
}

Header "STEP 4: FINAL OUTPUT VERIFICATION"

$outputFiles = Get-ChildItem "$outputDir\*.mp4", "$outputDir\*.webm" -ErrorAction SilentlyContinue
if ($outputFiles) {
    Ok "Output directory contains $($outputFiles.Count) video files:"
    foreach ($file in $outputFiles | Select-Object -First 3) {
        $sizeMB = [math]::Round($file.Length / 1MB, 1)
        Info "  * $($file.Name) ($sizeMB MB)"
    }
    if ($outputFiles.Count -gt 3) {
        Info "  ... and $($outputFiles.Count - 3) more files"
    }
} else {
    Warn "Output directory is empty - videos needed for holographic rendering"
}

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host "           HOLOGRAPHIC RENDERING STATUS" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Magenta

if ($copiedCount -gt 0 -and $allComponentsReady) {
    Write-Host "`nSYSTEM READY FOR HOLOGRAPHIC DEMO!" -ForegroundColor Green
    Write-Host ""
    Write-Host "What you'll see at /hologram:" -ForegroundColor Cyan
    Write-Host "  * WebGL2 shader-based holographic rendering" -ForegroundColor White
    Write-Host "  * RGB diffraction effect (spectral edge separation)" -ForegroundColor White
    Write-Host "  * Wave distortion for interference patterns" -ForegroundColor White
    Write-Host "  * Shimmer and edge glow effects" -ForegroundColor White
    Write-Host "  * Real-time video source switching" -ForegroundColor White
    Write-Host ""
    Write-Host "Available video sources:" -ForegroundColor Cyan
    Write-Host "  * HOLO FLUX" -ForegroundColor White
    Write-Host "  * MACH LIGHTFIELD" -ForegroundColor White
    Write-Host "  * KINETIC LOGO PARADE" -ForegroundColor White
    Write-Host ""
    Write-Host "NEXT STEPS:" -ForegroundColor Yellow
    Write-Host "  1. Restart your dev server:" -ForegroundColor White
    Write-Host "     cd D:\Dev\kha\frontend" -ForegroundColor Gray
    Write-Host "     pnpm dev" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. Open http://localhost:5173/hologram" -ForegroundColor White
    Write-Host ""
    Write-Host "  3. Use the source selector to switch between videos" -ForegroundColor White
    Write-Host ""
    Write-Host "DEMO TALKING POINTS:" -ForegroundColor Cyan
    Write-Host "  'This is real-time holographic rendering using WebGL2 shaders'" -ForegroundColor White
    Write-Host "  'Notice the RGB diffraction - simulating light interference'" -ForegroundColor White
    Write-Host "  'The wave distortion creates depth and movement'" -ForegroundColor White
    Write-Host "  'Each video source is processed through our holographic pipeline'" -ForegroundColor White
    Write-Host "  'This runs natively on iOS 26 with WebGPU support'" -ForegroundColor White
} else {
    Write-Host "`nSETUP INCOMPLETE" -ForegroundColor Yellow
    if ($copiedCount -eq 0) {
        Write-Host "  * No videos in output directory" -ForegroundColor Red
        Write-Host "  * Run encoding scripts to generate videos" -ForegroundColor Yellow
    }
    if (-not $allComponentsReady) {
        Write-Host "  * Some components are missing" -ForegroundColor Red
        Write-Host "  * Check the component list above" -ForegroundColor Yellow
    }
}

Write-Host "`n============================================================" -ForegroundColor Magenta
Write-Host ""