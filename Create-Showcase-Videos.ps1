# PowerShell Script for Creating Showcase Videos
# MACH LIGHTFIELD & KINETIC LOGO PARADE

param(
    [Parameter(Mandatory=$false)]
    [switch]$CreateVideos,
    [Parameter(Mandatory=$false)]
    [switch]$ConvertWebM
)

$ErrorActionPreference = "Stop"
Set-Location "D:\Dev\kha"

Write-Host @"

==================================================
    iRis SHOWCASE VIDEO CREATOR
    MACH LIGHTFIELD & KINETIC LOGO PARADE
==================================================
"@ -ForegroundColor Cyan

# Video specifications
$showcaseVideos = @(
    @{
        Name = "mach_lightfield_showcase"
        Look = "MACH LIGHTFIELD"
        Duration = "12-15s"
        Settings = @"
• Source Panel → MACH LIGHTFIELD
• Exposure: +0.15
• Ghost-fade: 0.25
• Orbit: Slow (one full turn in ~8-10s)
• Plan: Plus (no watermark)
"@
        Purpose = "Eye-candy physics look for main demo"
    },
    @{
        Name = "kinetic_logo_parade"
        Look = "KINETIC LOGO PARADE"
        Duration = "10-12s"
        Settings = @"
• Source Panel → KINETIC LOGO PARADE
• Logo: Load your iRis/TORI mark (PNG/SVG)
• Logo size: 0.9
• Trail length: 0.6
• Beat: 120 bpm (steady loop)
• Plan: Plus (no watermark)
"@
        Purpose = "Brand-ready logo animation"
    }
)

function Show-VideoInstructions {
    foreach ($video in $showcaseVideos) {
        Write-Host "`n===================================================" -ForegroundColor Magenta
        Write-Host " VIDEO: $($video.Name).mp4" -ForegroundColor Cyan
        Write-Host " Look: $($video.Look)" -ForegroundColor White
        Write-Host " Duration: $($video.Duration)" -ForegroundColor White
        Write-Host "===================================================" -ForegroundColor Magenta
        Write-Host "`nSettings:" -ForegroundColor Yellow
        Write-Host $video.Settings -ForegroundColor White
        Write-Host "`nPurpose: $($video.Purpose)" -ForegroundColor Gray
        Write-Host "`nSave to: site\showcase\$($video.Name).mp4" -ForegroundColor Green
    }
}

function Start-RecordingSession {
    Write-Host "`nStarting Recording Session..." -ForegroundColor Yellow
    
    # Open hologram studio
    Start-Process "http://localhost:5173/hologram-studio"
    Write-Host "  ✓ Opened hologram studio" -ForegroundColor Green
    
    Write-Host "`nRecording Steps:" -ForegroundColor Cyan
    Write-Host @"
    
1. MACH LIGHTFIELD:
   - Select MACH LIGHTFIELD from source panel
   - Set Exposure to +0.15
   - Set Ghost-fade to 0.25
   - Enable slow orbit
   - Press Start (Space key)
   - Record 12-15 seconds
   - File saves as iris_*.webm
   
2. KINETIC LOGO PARADE:
   - Load your logo (PNG/SVG)
   - Select KINETIC LOGO PARADE
   - Set Logo size to 0.9
   - Set Trail length to 0.6
   - Set Beat to 120 bpm
   - Press Start (Space key)
   - Record 10-12 seconds
   - File saves as iris_*.webm

"@ -ForegroundColor White
}

function Convert-ShowcaseVideos {
    Write-Host "`nConverting WebM to MP4..." -ForegroundColor Yellow
    
    $downloads = "$env:USERPROFILE\Downloads"
    $showcaseDir = "D:\Dev\kha\site\showcase"
    
    # Find recent iris_*.webm files
    $webmFiles = Get-ChildItem -Path $downloads -Filter "iris_*.webm" -ErrorAction SilentlyContinue | 
                 Sort-Object LastWriteTime -Descending | 
                 Select-Object -First 2
    
    if ($webmFiles.Count -eq 0) {
        Write-Host "  No iris_*.webm files found in Downloads" -ForegroundColor Red
        return
    }
    
    Write-Host "  Found $($webmFiles.Count) WebM file(s)" -ForegroundColor Cyan
    
    foreach ($file in $webmFiles) {
        Write-Host "`n  Processing: $($file.Name)" -ForegroundColor White
        Write-Host "  Which video is this?" -ForegroundColor Yellow
        Write-Host "    [1] MACH LIGHTFIELD" -ForegroundColor Cyan
        Write-Host "    [2] KINETIC LOGO PARADE" -ForegroundColor Cyan
        Write-Host "    [S] Skip" -ForegroundColor Gray
        
        $choice = Read-Host "  Your choice"
        
        $outputName = switch ($choice) {
            "1" { "mach_lightfield_showcase.mp4" }
            "2" { "kinetic_logo_parade.mp4" }
            default { $null }
        }
        
        if ($outputName) {
            $outputPath = Join-Path $showcaseDir $outputName
            
            if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
                Write-Host "  Converting to MP4..." -ForegroundColor Yellow
                ffmpeg -y -i $file.FullName `
                    -c:v libx264 `
                    -preset slow `
                    -crf 22 `
                    -pix_fmt yuv420p `
                    -movflags +faststart `
                    -c:a aac `
                    -b:a 128k `
                    $outputPath 2>$null
                    
                if (Test-Path $outputPath) {
                    $sizeMB = [math]::Round((Get-Item $outputPath).Length / 1MB, 2)
                    Write-Host "  ✓ Saved: $outputName (${sizeMB}MB)" -ForegroundColor Green
                }
            } else {
                Write-Host "  ✗ FFmpeg not found" -ForegroundColor Red
                Write-Host "  Install from: https://ffmpeg.org/download.html" -ForegroundColor Yellow
            }
        }
    }
}

function Check-ShowcaseStatus {
    Write-Host "`nChecking Showcase Videos..." -ForegroundColor Yellow
    
    $showcaseDir = "D:\Dev\kha\site\showcase"
    $requiredVideos = @(
        "mach_lightfield_showcase.mp4",
        "kinetic_logo_parade.mp4"
    )
    
    foreach ($video in $requiredVideos) {
        $path = Join-Path $showcaseDir $video
        if (Test-Path $path) {
            $file = Get-Item $path
            $sizeMB = [math]::Round($file.Length / 1MB, 2)
            $modified = $file.LastWriteTime.ToString("yyyy-MM-dd HH:mm")
            Write-Host "  ✓ $video - ${sizeMB}MB - Modified: $modified" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $video - Not created yet" -ForegroundColor Yellow
        }
    }
}

# Main execution
if ($CreateVideos) {
    Show-VideoInstructions
    Start-RecordingSession
    return
}

if ($ConvertWebM) {
    Convert-ShowcaseVideos
    return
}

# Interactive menu
do {
    Write-Host "`n=== SHOWCASE VIDEO MENU ===" -ForegroundColor Cyan
    Write-Host "  [1] View recording instructions" -ForegroundColor White
    Write-Host "  [2] Start recording session" -ForegroundColor White
    Write-Host "  [3] Convert WebM to MP4" -ForegroundColor White
    Write-Host "  [4] Check video status" -ForegroundColor White
    Write-Host "  [Q] Quit" -ForegroundColor Gray
    
    $choice = Read-Host "`nYour choice"
    
    switch ($choice) {
        "1" { Show-VideoInstructions }
        "2" { Start-RecordingSession }
        "3" { Convert-ShowcaseVideos }
        "4" { Check-ShowcaseStatus }
        "q" { break }
        default { Write-Host "  Invalid choice" -ForegroundColor Red }
    }
} while ($choice -ne "q")

Write-Host "`n✨ Happy recording!" -ForegroundColor Cyan