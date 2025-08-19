# Social-Export-Pipeline.ps1
# Complete pipeline: HDR source → SDR → Social (TikTok/Snap)
param(
    [Parameter(Mandatory=$true)]
    [string]$InputVideo,
    
    [switch]$SkipSDR,
    [switch]$SkipSocial,
    [switch]$OpenBrowser
)

Write-Host @"
╔════════════════════════════════════════════╗
║       SOCIAL EXPORT PIPELINE              ║
║   HDR → SDR → TikTok/Snapchat Ready       ║
╚════════════════════════════════════════════╝
"@ -ForegroundColor Magenta

$ErrorActionPreference = 'Stop'

# Paths
$wowpackInput = "D:\Dev\kha\content\wowpack\input"
$socialInput = "D:\Dev\kha\content\socialpack\input"
$basename = [System.IO.Path]::GetFileNameWithoutExtension($InputVideo)

Write-Host "`nProcessing: $basename" -ForegroundColor Cyan

# Step 1: Generate SDR if needed
if (-not $SkipSDR) {
    Write-Host "`n[1/3] Creating SDR version..." -ForegroundColor Yellow
    
    # Copy input to wowpack input folder
    Copy-Item $InputVideo "$wowpackInput\$([System.IO.Path]::GetFileName($InputVideo))" -Force
    
    # Run WowPack to create SDR
    & "D:\Dev\kha\tools\encode\Build-WowPack.ps1" `
        -InputFile $InputVideo `
        -Basename $basename `
        -DoSDR `
        -Framerate 60
    
    $sdrFile = "D:\Dev\kha\content\wowpack\video\sdr\${basename}_sdr.mp4"
    
    if (Test-Path $sdrFile) {
        Write-Host "  ✓ SDR created: $sdrFile" -ForegroundColor Green
        
        # Copy SDR to social input
        Copy-Item $sdrFile $socialInput -Force
    } else {
        Write-Host "  ✗ SDR creation failed!" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[1/3] Skipping SDR (using existing)" -ForegroundColor Gray
    $sdrFile = "$socialInput\${basename}_sdr.mp4"
}

# Step 2: Create social exports
if (-not $SkipSocial) {
    Write-Host "`n[2/3] Creating social exports..." -ForegroundColor Yellow
    
    & "D:\Dev\kha\tools\encode\Build-SocialPack.ps1" `
        -Input $sdrFile `
        -Framerate 60 `
        -VideoBitrateMbps 10 `
        -MaxFileSizeMB 250
    
    Write-Host "  ✓ Social exports complete!" -ForegroundColor Green
}

# Step 3: Show results
Write-Host "`n[3/3] Export Summary:" -ForegroundColor Cyan

$outputs = @{
    "TikTok" = "D:\Dev\kha\content\socialpack\out\tiktok\${basename}_tiktok_1080x1920.mp4"
    "Snapchat" = "D:\Dev\kha\content\socialpack\out\snap\${basename}_snap_1080x1920.mp4"
    "Thumbnails" = "D:\Dev\kha\content\socialpack\thumbs\$basename\"
}

foreach ($platform in $outputs.Keys) {
    if (Test-Path $outputs[$platform]) {
        $size = if ($platform -ne "Thumbnails") {
            $mb = [math]::Round((Get-Item $outputs[$platform]).Length / 1MB, 2)
            " ($mb MB)"
        } else { "" }
        Write-Host "  ✓ $platform$size" -ForegroundColor Green
        Write-Host "    $($outputs[$platform])" -ForegroundColor Gray
    }
}

# Get LAN IP for testing
$lanIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Wi-Fi", "Ethernet" | 
          Where-Object {$_.IPAddress -like "192.168.*" -or $_.IPAddress -like "10.*"} | 
          Select-Object -First 1).IPAddress

if ($lanIP) {
    Write-Host "`nTest on phone/tablet:" -ForegroundColor Yellow
    Write-Host "  TikTok:  http://${lanIP}:3000/social/tiktok/${basename}_tiktok_1080x1920.mp4" -ForegroundColor Cyan
    Write-Host "  Snap:    http://${lanIP}:3000/social/snap/${basename}_snap_1080x1920.mp4" -ForegroundColor Cyan
}

if ($OpenBrowser) {
    Start-Process "http://localhost:5173/social"
}

Write-Host "`n╔════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host   "║     READY FOR SOCIAL MEDIA!               ║" -ForegroundColor Green
Write-Host   "║                                            ║" -ForegroundColor Green
Write-Host   "║  Next steps:                               ║" -ForegroundColor Green
Write-Host   "║  1. Transfer to phone (AirDrop/Drive)     ║" -ForegroundColor Green
Write-Host   "║  2. Upload to TikTok/Snapchat             ║" -ForegroundColor Green
Write-Host   "║  3. Add trending music & hashtags         ║" -ForegroundColor Green
Write-Host   "║  4. Post at peak times (6-10am, 7-11pm)   ║" -ForegroundColor Green
Write-Host   "╚════════════════════════════════════════════╝" -ForegroundColor Green