# Quick Video Creation Helper Script
Write-Host "🎬 iRis Video Creator Helper" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

# Check if dev server is running
$devServerRunning = Test-NetConnection -ComputerName localhost -Port 5173 -InformationLevel Quiet
if (-not $devServerRunning) {
    Write-Host "⚠️  Dev server not running. Starting it now..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd D:\Dev\kha\tori_ui_svelte; pnpm dev" -WindowStyle Normal
    Write-Host "Waiting for server to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}

# Function to simulate different plans
function Set-Plan {
    param([string]$Plan)
    
    $js = switch($Plan) {
        'free' { "localStorage.setItem('iris.plan', 'free'); location.reload();" }
        'plus' { "localStorage.setItem('iris.plan', 'plus'); location.reload();" }
        'pro' { "localStorage.setItem('iris.plan', 'pro'); location.reload();" }
    }
    
    Write-Host "Plan set to: $Plan" -ForegroundColor Green
    Write-Host "Run this in browser console (F12):" -ForegroundColor Yellow
    Write-Host $js -ForegroundColor White
    Set-Clipboard -Value $js
    Write-Host "✅ JavaScript copied to clipboard!" -ForegroundColor Green
}

# Menu
Write-Host "`n📋 QUICK ACTIONS:" -ForegroundColor Magenta
Write-Host "1. Open Hologram Studio"
Write-Host "2. Set FREE plan (10s, watermark)"
Write-Host "3. Set PLUS plan (60s, no watermark)"
Write-Host "4. Set PRO plan (300s, no watermark)"
Write-Host "5. Convert WebM to MP4"
Write-Host "6. Create dummy export files"
Write-Host "7. Open showcase folder"
Write-Host "8. View Video Creation Guide"

$choice = Read-Host "`nSelect action (1-8)"

switch ($choice) {
    '1' {
        Start-Process "http://localhost:5173/hologram"
        Write-Host "✅ Opened Hologram Studio in browser" -ForegroundColor Green
    }
    '2' {
        Set-Plan -Plan 'free'
    }
    '3' {
        Set-Plan -Plan 'plus'
    }
    '4' {
        Set-Plan -Plan 'pro'
    }
    '5' {
        $webmFile = Read-Host "Enter WebM file path (or drag & drop)"
        $webmFile = $webmFile.Trim('"')
        if (Test-Path $webmFile) {
            & "D:\Dev\kha\tools\exporters\webm-to-mp4.ps1" -In $webmFile
        } else {
            Write-Host "File not found: $webmFile" -ForegroundColor Red
        }
    }
    '6' {
        # Create dummy export files for demonstration
        Write-Host "Creating dummy export files..." -ForegroundColor Yellow
        
        # Create directories if they don't exist
        @(
            "D:\Dev\kha\exports\video",
            "D:\Dev\kha\exports\models",
            "D:\Dev\kha\exports\textures_ktx2"
        ) | ForEach-Object {
            if (-not (Test-Path $_)) {
                New-Item -ItemType Directory -Path $_ -Force | Out-Null
            }
        }
        
        # Create dummy files
        @(
            "D:\Dev\kha\exports\video\iris_export_001.mp4",
            "D:\Dev\kha\exports\video\iris_export_001_prores.mov",
            "D:\Dev\kha\exports\video\sponsor_ready_clip.mp4",
            "D:\Dev\kha\exports\models\hologram_asset.glb",
            "D:\Dev\kha\exports\textures_ktx2\texture_001.ktx2",
            "D:\Dev\kha\exports\textures_ktx2\normal_map.ktx2"
        ) | ForEach-Object {
            "dummy content" | Out-File $_ -Force
            Write-Host "  Created: $_" -ForegroundColor Gray
        }
        
        Write-Host "✅ Dummy files created!" -ForegroundColor Green
    }
    '7' {
        Start-Process explorer "D:\Dev\kha\site\showcase"
        Write-Host "✅ Opened showcase folder" -ForegroundColor Green
    }
    '8' {
        Start-Process notepad "D:\Dev\kha\VIDEO_CREATION_GUIDE.md"
        Write-Host "✅ Opened Video Creation Guide" -ForegroundColor Green
    }
    default {
        Write-Host "Invalid choice" -ForegroundColor Red
    }
}

Write-Host "`n💡 TIP: Use Windows Game Bar (Win+G) as backup for screen recording" -ForegroundColor Yellow
Write-Host "💡 TIP: The browser recorder saves to Downloads folder by default" -ForegroundColor Yellow
