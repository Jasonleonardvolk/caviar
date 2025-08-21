# PowerShell Script for Creating Marketing Videos
# iRis Phase 4 - Content Creation Helper

param(
    [Parameter(Mandatory=$false)]
    [switch]$CreateDummyFiles,
    [Parameter(Mandatory=$false)]
    [switch]$OpenStudio
)

$ErrorActionPreference = "Stop"
Set-Location "D:\Dev\kha"

Write-Host @"

==================================================
    iRis PHASE 4 - MARKETING VIDEO CREATION
==================================================
"@ -ForegroundColor Cyan

# Video specifications
$videos = @(
    @{
        Name = "A_shock_proof"
        Duration = "10s"
        Plan = "Free"
        Description = "Visual demo with watermark (shock value)"
        Script = @"
1. Open hologram studio
2. Select high-contrast/neon preset
3. Click 'Record 10s' (Free plan)
4. Let it auto-stop at 10s
5. Show watermark clearly in video
6. Message: 'Free gets you started'
"@
    },
    @{
        Name = "B_how_to_60s"
        Duration = "20-30s"
        Plan = "Plus"
        Description = "Tutorial showing the process"
        Script = @"
1. Start on /pricing page
2. Click 'Get Plus'
3. Complete checkout (test card)
4. Return to hologram studio
5. Show 'Plus' badge
6. Click 'Record 60s'
7. Record 20-30s demo
8. Stop and download
9. Show NO watermark
"@
    },
    @{
        Name = "C_buyers_clip"
        Duration = "15-20s"
        Plan = "Plus/Pro"
        Description = "Show deliverables & exports"
        Script = @"
1. Open File Explorer
2. Navigate to D:\Dev\kha\exports\video\
3. Show multiple .mp4 files (5s)
4. Double-click one to play (5s)
5. Show professional quality
6. End card: 'Professional exports'
"@
    }
)

# Main Menu
function Show-Menu {
    Write-Host "`nSelect an option:" -ForegroundColor Yellow
    Write-Host "  [1] Open Hologram Studio (start recording)" -ForegroundColor Cyan
    Write-Host "  [2] Show video scripts & instructions" -ForegroundColor Cyan
    Write-Host "  [3] Convert WebM to MP4" -ForegroundColor Cyan
    Write-Host "  [4] Check recorded videos status" -ForegroundColor Cyan
    Write-Host "  [5] Copy to final locations" -ForegroundColor Cyan
    Write-Host "  [6] Create dummy export files (for demo)" -ForegroundColor Cyan
    Write-Host "  [7] Open screen recorder instructions" -ForegroundColor Cyan
    Write-Host "  [Q] Quit" -ForegroundColor Gray
}

# Function to show video scripts
function Show-Scripts {
    foreach ($video in $videos) {
        Write-Host "`n===================================================" -ForegroundColor Magenta
        Write-Host " VIDEO: $($video.Name).mp4" -ForegroundColor Cyan
        Write-Host " Duration: $($video.Duration) | Plan: $($video.Plan)" -ForegroundColor White
        Write-Host "===================================================" -ForegroundColor Magenta
        Write-Host $video.Script -ForegroundColor White
    }
}

# Function to convert WebM to MP4
function Convert-Video {
    $webmFile = Read-Host "`nEnter WebM file path (or drag & drop)"
    $webmFile = $webmFile.Trim('"')
    
    if (-not (Test-Path $webmFile)) {
        Write-Host "  ✗ File not found: $webmFile" -ForegroundColor Red
        return
    }
    
    $outputFile = $webmFile -replace '\.webm$', '.mp4'
    
    Write-Host "  Converting to MP4..." -ForegroundColor Yellow
    
    # Check if ffmpeg exists
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        ffmpeg -y -i "$webmFile" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$outputFile"
        Write-Host "  ✓ Converted to: $outputFile" -ForegroundColor Green
    } else {
        Write-Host "  ✗ FFmpeg not found. Install from: https://ffmpeg.org/download.html" -ForegroundColor Red
    }
}

# Function to check video status
function Check-Status {
    Write-Host "`nChecking video status..." -ForegroundColor Yellow
    
    $locations = @(
        "D:\Dev\kha\site\showcase\A_shock_proof.mp4",
        "D:\Dev\kha\site\showcase\B_how_to_60s.mp4",
        "D:\Dev\kha\site\showcase\C_buyers_clip.mp4"
    )
    
    foreach ($path in $locations) {
        if (Test-Path $path) {
            $file = Get-Item $path
            $sizeMB = [math]::Round($file.Length / 1MB, 2)
            Write-Host "  ✓ $(Split-Path $path -Leaf) - ${sizeMB}MB" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $(Split-Path $path -Leaf) - Not created yet" -ForegroundColor Yellow
        }
    }
    
    # Check for any iris_*.webm files in Downloads
    $downloads = "$env:USERPROFILE\Downloads"
    $irisFiles = Get-ChildItem -Path $downloads -Filter "iris_*.webm" -ErrorAction SilentlyContinue
    
    if ($irisFiles) {
        Write-Host "`n  Found recordings in Downloads:" -ForegroundColor Cyan
        foreach ($file in $irisFiles) {
            $sizeMB = [math]::Round($file.Length / 1MB, 2)
            Write-Host "    - $($file.Name) (${sizeMB}MB)" -ForegroundColor Gray
        }
    }
}

# Function to copy videos to final locations
function Copy-Videos {
    Write-Host "`nCopying videos to final locations..." -ForegroundColor Yellow
    
    # Map source to destination
    $videoMap = @{
        "A" = "A_shock_proof.mp4"
        "B" = "B_how_to_60s.mp4"
        "C" = "C_buyers_clip.mp4"
    }
    
    foreach ($key in $videoMap.Keys) {
        $fileName = $videoMap[$key]
        Write-Host "`n  Video $key: $fileName" -ForegroundColor Cyan
        
        $source = Read-Host "    Enter source file path (or press Enter to skip)"
        if ($source -and (Test-Path $source.Trim('"'))) {
            $source = $source.Trim('"')
            
            # Copy to showcase folder
            $dest1 = "D:\Dev\kha\site\showcase\$fileName"
            Copy-Item $source $dest1 -Force
            Write-Host "    ✓ Copied to: $dest1" -ForegroundColor Green
            
            # Note about Google Drive
            Write-Host "    ℹ Also copy to: Drive\iRis\Showcase\$fileName" -ForegroundColor Yellow
        }
    }
}

# Function to create dummy export files
function Create-DummyExports {
    Write-Host "`nCreating dummy export files for demo..." -ForegroundColor Yellow
    
    $exportDir = "D:\Dev\kha\exports\video"
    if (-not (Test-Path $exportDir)) {
        New-Item -ItemType Directory -Path $exportDir -Force | Out-Null
    }
    
    $dummyFiles = @(
        "hologram_export_001.mp4",
        "hologram_export_002.mp4",
        "iris_professional_4K.mp4",
        "iris_studio_quality.mp4",
        "client_deliverable_final.mp4"
    )
    
    foreach ($file in $dummyFiles) {
        $path = Join-Path $exportDir $file
        # Create a small dummy file
        "iRis Professional Export - $file" | Out-File $path -Encoding UTF8
        Write-Host "  ✓ Created: $file" -ForegroundColor Green
    }
    
    Write-Host "`n  Dummy files created in: $exportDir" -ForegroundColor Cyan
    Write-Host "  Use these for Video C (buyers clip)" -ForegroundColor Yellow
}

# Function to show screen recording instructions
function Show-ScreenRecording {
    Write-Host @"

==================================================
         SCREEN RECORDING INSTRUCTIONS
==================================================

WINDOWS (Built-in):
1. Press Win + G to open Game Bar
2. Click Record button (or Win + Alt + R)
3. Perform your demo
4. Stop with Win + Alt + R
5. Find in: Videos\Captures

OBS STUDIO (Professional):
1. Download: https://obsproject.com
2. Add Source → Display Capture
3. Set to 1920x1080, 30fps
4. Record to MP4
5. Better quality than Game Bar

IPHONE RECORDING:
1. Settings → Control Center → Add Screen Recording
2. Open site in Safari
3. Swipe down → Tap record button
4. Do demo → Stop recording
5. AirDrop or cable to PC

QUICK TIPS:
- Clean desktop before recording
- Close unnecessary apps
- Use consistent mouse movements
- Record in 1920x1080 if possible
- Keep videos under 60 seconds

"@ -ForegroundColor White
}

# Open hologram studio if requested
if ($OpenStudio) {
    Start-Process "http://localhost:5173/hologram-studio"
    Write-Host "✓ Opened hologram studio" -ForegroundColor Green
}

# Create dummy files if requested
if ($CreateDummyFiles) {
    Create-DummyExports
}

# Main loop
do {
    Show-Menu
    $choice = Read-Host "`nYour choice"
    
    switch ($choice) {
        "1" { 
            Start-Process "http://localhost:5173/hologram-studio"
            Write-Host "  ✓ Opened hologram studio" -ForegroundColor Green
        }
        "2" { Show-Scripts }
        "3" { Convert-Video }
        "4" { Check-Status }
        "5" { Copy-Videos }
        "6" { Create-DummyExports }
        "7" { Show-ScreenRecording }
        "q" { break }
        default { Write-Host "  Invalid choice" -ForegroundColor Red }
    }
} while ($choice -ne "q")

Write-Host "`n✨ Good luck with your video creation!" -ForegroundColor Cyan