# PowerShell Launch Day Script
# iRis Phase 5 - Minimum Viable Blast

param(
    [Parameter(Mandatory=$false)]
    [switch]$TestMode,
    [Parameter(Mandatory=$false)]
    [switch]$UpdateLanding,
    [Parameter(Mandatory=$false)]
    [switch]$CheckStatus
)

$ErrorActionPreference = "Stop"
Set-Location "D:\Dev\kha"

Write-Host @"

==================================================
       iRis PHASE 5 - LAUNCH DAY 🚀
==================================================
"@ -ForegroundColor Cyan

# Function to check launch readiness
function Test-LaunchReadiness {
    Write-Host "`nChecking Launch Readiness..." -ForegroundColor Yellow
    
    $checks = @(
        @{Name="Landing page"; Path="tori_ui_svelte\src\routes\+page.svelte.new"}
        @{Name="Video A"; Path="site\showcase\A_shock_proof.mp4"}
        @{Name="Video B"; Path="site\showcase\B_how_to_60s.mp4"}
        @{Name="Video C"; Path="site\showcase\C_buyers_clip.mp4"}
        @{Name="Challenge page"; Path="site\challenge\index.md"}
        @{Name="Founders terms"; Path="docs\legal\founders100_terms.md"}
    )
    
    $ready = $true
    foreach ($check in $checks) {
        if (Test-Path $check.Path) {
            Write-Host "  ✓ $($check.Name)" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $($check.Name) missing" -ForegroundColor Red
            $ready = $false
        }
    }
    
    # Check if server is running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        Write-Host "  ✓ Dev server running" -ForegroundColor Green
    } catch {
        Write-Host "  ⚠ Dev server not running" -ForegroundColor Yellow
    }
    
    return $ready
}

# Function to update landing page
function Update-LandingPage {
    Write-Host "`nUpdating Landing Page..." -ForegroundColor Yellow
    
    $source = "tori_ui_svelte\src\routes\+page.svelte.new"
    $dest = "tori_ui_svelte\src\routes\+page.svelte"
    $backup = "tori_ui_svelte\src\routes\+page.svelte.backup_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    
    if (Test-Path $source) {
        # Backup current
        if (Test-Path $dest) {
            Copy-Item $dest $backup -Force
            Write-Host "  ✓ Backed up current page to $(Split-Path $backup -Leaf)" -ForegroundColor Green
        }
        
        # Copy new
        Copy-Item $source $dest -Force
        Write-Host "  ✓ Landing page updated" -ForegroundColor Green
        
        # Prompt to commit
        $commit = Read-Host "`n  Commit changes? (y/n)"
        if ($commit -eq 'y') {
            Set-Location "tori_ui_svelte"
            git add .
            git commit -m "Launch: New landing page with hero CTAs"
            Write-Host "  ✓ Changes committed" -ForegroundColor Green
            
            $push = Read-Host "  Push to origin? (y/n)"
            if ($push -eq 'y') {
                git push origin main
                Write-Host "  ✓ Pushed to GitHub" -ForegroundColor Green
            }
            Set-Location ".."
        }
    } else {
        Write-Host "  ✗ New landing page not found" -ForegroundColor Red
    }
}

# Function to generate social posts
function Generate-SocialPosts {
    Write-Host "`nGenerating Social Posts..." -ForegroundColor Yellow
    
    $posts = @{
        "Twitter" = @"
We killed the 45° glass. You can't mix two light fields without optics—but you can make footage that looks like you did.

iRis: Creator-Pro Hologram Studio
Record in minutes. Own the master.

🚀 #HologramDrop challenge LIVE
🎁 First 100 get Founders pricing

Try free → https://your-app.vercel.app
"@
        
        "Instagram" = @"
Make 'hologram' videos brands pay for 🔮

✨ Physics-native looks
⚡ 10s free exports  
🚀 No watermark with Plus

Join #HologramDrop - win 1 year Pro!

Link in bio 👆
"@
        
        "TikTok" = @"
POV: You just made a sponsor-ready hologram in 60 seconds

Free 10s export TODAY only 
#HologramDrop #CreatorTools #HologramStudio
"@
    }
    
    # Save to files
    $socialDir = "site\social_posts"
    if (-not (Test-Path $socialDir)) {
        New-Item -ItemType Directory -Path $socialDir -Force | Out-Null
    }
    
    foreach ($platform in $posts.Keys) {
        $content = $posts[$platform]
        $filename = "$socialDir\$platform`_post.txt"
        $content | Out-File $filename -Encoding UTF8
        Write-Host "  ✓ $platform post saved" -ForegroundColor Green
    }
    
    Write-Host "`n  Posts saved to: $socialDir" -ForegroundColor Cyan
}

# Function to open launch tools
function Open-LaunchTools {
    Write-Host "`nOpening Launch Tools..." -ForegroundColor Yellow
    
    # Open key pages
    Start-Process "http://localhost:5173"
    Start-Process "http://localhost:5173/hologram-studio"
    Start-Process "http://localhost:5173/pricing"
    
    # Open social media
    Start-Process "https://twitter.com/compose/tweet"
    Start-Process "https://www.instagram.com"
    
    # Open Stripe dashboard
    Start-Process "https://dashboard.stripe.com/test/payments"
    
    Write-Host "  ✓ Opened all launch tools" -ForegroundColor Green
}

# Function to create launch report
function Create-LaunchReport {
    $report = @{
        Date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Checklist = @{
            LandingPage = Test-Path "tori_ui_svelte\src\routes\+page.svelte.new"
            Videos = (Test-Path "site\showcase\A_shock_proof.mp4") -and 
                    (Test-Path "site\showcase\B_how_to_60s.mp4") -and 
                    (Test-Path "site\showcase\C_buyers_clip.mp4")
            Challenge = Test-Path "site\challenge\index.md"
            Founders = Test-Path "docs\legal\founders100_terms.md"
        }
        URLs = @{
            Production = "https://your-app.vercel.app"
            HologramStudio = "https://your-app.vercel.app/hologram-studio"
            Pricing = "https://your-app.vercel.app/pricing"
            Challenge = "https://your-app.vercel.app/challenge"
        }
    }
    
    $report | ConvertTo-Json -Depth 3 | Out-File "launch_report.json" -Encoding UTF8
    Write-Host "`n  Launch report saved to launch_report.json" -ForegroundColor Cyan
}

# Main menu
if ($CheckStatus) {
    Test-LaunchReadiness
    Create-LaunchReport
    exit
}

if ($UpdateLanding) {
    Update-LandingPage
    exit
}

# Interactive launch menu
do {
    Write-Host "`n=== LAUNCH MENU ===" -ForegroundColor Cyan
    Write-Host "  [1] Check launch readiness" -ForegroundColor White
    Write-Host "  [2] Update landing page" -ForegroundColor White
    Write-Host "  [3] Generate social posts" -ForegroundColor White
    Write-Host "  [4] Open launch tools" -ForegroundColor White
    Write-Host "  [5] View launch checklist" -ForegroundColor White
    Write-Host "  [6] Create launch report" -ForegroundColor White
    Write-Host "  [Q] Quit" -ForegroundColor Gray
    
    $choice = Read-Host "`nYour choice"
    
    switch ($choice) {
        "1" { 
            $ready = Test-LaunchReadiness
            if ($ready) {
                Write-Host "`n✅ Ready to launch!" -ForegroundColor Green
            } else {
                Write-Host "`n⚠ Not ready - fix issues above" -ForegroundColor Yellow
            }
        }
        "2" { Update-LandingPage }
        "3" { Generate-SocialPosts }
        "4" { Open-LaunchTools }
        "5" { 
            if (Test-Path "PHASE_5_LAUNCH_CHECKLIST.md") {
                notepad "PHASE_5_LAUNCH_CHECKLIST.md"
            }
        }
        "6" { Create-LaunchReport }
        "q" { break }
        default { Write-Host "  Invalid choice" -ForegroundColor Red }
    }
} while ($choice -ne "q")

Write-Host "`n🚀 Good luck with your launch!" -ForegroundColor Cyan