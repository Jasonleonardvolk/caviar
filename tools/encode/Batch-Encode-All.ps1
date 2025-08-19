# Batch-Encode-All.ps1
# Encodes all three WOW Pack hero videos in one go

$ErrorActionPreference = "Stop"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     WOW Pack Batch Encoder                    " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Change to encode directory
$encodePath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $encodePath

# Define all videos to encode
$videos = @(
    @{
        name = "holo_flux_loop"
        fps = 60
        sdr = $true
        hls = $false
        description = "Holographic flux animation loop"
    },
    @{
        name = "mach_lightfield"
        fps = 60
        sdr = $true
        hls = $false
        description = "Mach lightfield visualization"
    },
    @{
        name = "kinetic_logo_parade"
        fps = 60
        sdr = $true
        hls = $true
        description = "Kinetic logo animation parade"
    }
)

$successCount = 0
$failedVideos = @()

Write-Host "Checking for video files..." -ForegroundColor Yellow
Write-Host ""

foreach($v in $videos) {
    $inputPath = "..\..\content\wowpack\input\$($v.name).mov"
    $altInputPath = "..\..\content\wowpack\input\$($v.name).mp4"
    
    $found = $false
    $actualInput = ""
    
    if(Test-Path $inputPath) {
        $found = $true
        $actualInput = $inputPath
        Write-Host "[OK] Found: $($v.name).mov" -ForegroundColor Green
    } elseif(Test-Path $altInputPath) {
        $found = $true
        $actualInput = $altInputPath
        Write-Host "[OK] Found: $($v.name).mp4" -ForegroundColor Green
    } else {
        Write-Host "[X] Missing: $($v.name).mov or .mp4" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Starting encoding process..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

foreach($v in $videos) {
    $inputPath = "..\..\content\wowpack\input\$($v.name).mov"
    $altInputPath = "..\..\content\wowpack\input\$($v.name).mp4"
    
    $actualInput = ""
    if(Test-Path $inputPath) {
        $actualInput = $inputPath
    } elseif(Test-Path $altInputPath) {
        $actualInput = $altInputPath
    }
    
    if($actualInput) {
        Write-Host "[$($successCount + 1)/3] Encoding: $($v.description)" -ForegroundColor Green
        Write-Host "    File: $actualInput" -ForegroundColor Gray
        Write-Host "    Settings: $($v.fps)fps" -NoNewline -ForegroundColor Gray
        if($v.sdr) { Write-Host ", SDR fallback" -NoNewline -ForegroundColor Gray }
        if($v.hls) { Write-Host ", HLS streaming" -NoNewline -ForegroundColor Gray }
        Write-Host ""
        
        try {
            # Build arguments properly
            $scriptPath = ".\Build-WowPack.ps1"
            $arguments = @{
                Basename = $v.name
                Input = $actualInput
                Framerate = $v.fps
            }
            
            # Add switches if needed
            if($v.sdr) { 
                & $scriptPath @arguments -DoSDR
            } elseif($v.hls) {
                & $scriptPath @arguments -DoSDR -MakeHLS
            } else {
                & $scriptPath @arguments
            }
            
            # Special case for kinetic_logo_parade with both SDR and HLS
            if($v.name -eq "kinetic_logo_parade" -and $v.sdr -and $v.hls) {
                & $scriptPath -Basename $v.name -Input $actualInput -Framerate $v.fps -DoSDR -MakeHLS
            } elseif($v.sdr -and -not $v.hls) {
                & $scriptPath -Basename $v.name -Input $actualInput -Framerate $v.fps -DoSDR
            }
            
            $successCount++
            Write-Host "    [OK] Success!" -ForegroundColor Green
            Write-Host ""
        } catch {
            Write-Host "    [FAIL] Failed: $_" -ForegroundColor Red
            $failedVideos += $v.name
            Write-Host ""
        }
    } else {
        Write-Host "[SKIP] Skipping $($v.name) - file not found" -ForegroundColor Yellow
        $failedVideos += $v.name
        Write-Host ""
    }
}

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Encoding Summary" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Successfully encoded: $successCount / $($videos.Count)" -ForegroundColor $(if($successCount -eq $videos.Count) {"Green"} else {"Yellow"})

if($failedVideos.Count -gt 0) {
    Write-Host "Failed or skipped:" -ForegroundColor Red
    foreach($failed in $failedVideos) {
        Write-Host "  - $failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Running verification..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

try {
    & ..\release\Verify-WowPack.ps1
} catch {
    Write-Host "Verification failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Batch encoding complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan

if($successCount -eq $videos.Count) {
    Write-Host ""
    Write-Host "*** All videos encoded successfully! ***" -ForegroundColor Green
    Write-Host ""
    Write-Host "Test your clips at:" -ForegroundColor Cyan
    Write-Host "  http://localhost:3000/hologram?clip=holo_flux_loop" -ForegroundColor White
    Write-Host "  http://localhost:3000/hologram?clip=mach_lightfield" -ForegroundColor White
    Write-Host "  http://localhost:3000/hologram?clip=kinetic_logo_parade" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "WARNING: Some videos were not encoded. Please check the missing files." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
