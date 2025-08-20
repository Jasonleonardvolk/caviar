param([string]$ProjectRoot = "D:\Dev\kha")

# ULTIMATE PRODUCTION READINESS CHECK
# This script validates EVERYTHING for tonight's launch

Write-Host "`n" -NoNewline
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║     🚀 CAVIAR/KHA ULTIMATE LAUNCH READINESS CHECK 🚀     ║" -ForegroundColor Cyan
Write-Host "║            Checking ALL systems for production            ║" -ForegroundColor Yellow
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Magenta

function Ok($m){Write-Host "[✅] $m" -f Green}
function Info($m){Write-Host "[ℹ️] $m" -f Cyan}
function Warn($m){Write-Host "[⚠️] $m" -f Yellow}
function Fail($m){Write-Host "[❌] $m" -f Red}
function Header($m){
    Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -f DarkMagenta
    Write-Host "  $m" -f Cyan
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -f DarkMagenta
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$allSystemsGo = $true
$criticalCount = 0
$warningCount = 0
$successCount = 0

# Master report object
$masterReport = @{
    timestamp = $timestamp
    systems = @{}
    metrics = @{}
    readiness = @{}
}

Header "SECTION 1: CORE HEALTH CHECK"

# 1.1 Dev Server Check
Write-Host "`n🌐 Dev Server Status:" -ForegroundColor Yellow
function TestPort($port) { 
    try { 
        $result = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$port/health/ping" -TimeoutSec 2 -ErrorAction SilentlyContinue
        return $result.StatusCode -eq 200 
    } catch { 
        return $false 
    } 
}

$ports = @(5173, 3000, 3310)
$activePort = $null
foreach($p in $ports) { 
    if(TestPort $p) { 
        $activePort = $p
        break 
    } 
}

if($activePort) {
    Ok "Dev server ACTIVE on port $activePort"
    $successCount++
    
    # Get detailed health
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$activePort/health" -TimeoutSec 5
        
        if($health.ok -eq $true) {
            Ok "Health endpoint: HEALTHY"
            $successCount++
        } else {
            Warn "Health endpoint: DEGRADED"
            $warningCount++
        }
        
        # Store health data
        $masterReport.systems.health = $health
        
        # Check specific components
        if($health.readiness.demo -eq $true) {
            Ok "Production mode: READY (demo=true)"
            $successCount++
        } else {
            Fail "Production mode: NOT READY"
            $criticalCount++
        }
        
        if($health.files.presentCount -eq $health.files.requiredTotal) {
            Ok "Required files: $($health.files.presentCount)/$($health.files.requiredTotal) present"
            $successCount++
        } else {
            Fail "Missing files: $($health.files.missing -join ', ')"
            $criticalCount++
        }
        
    } catch {
        Fail "Could not get health details: $_"
        $criticalCount++
    }
} else {
    Fail "Dev server NOT RUNNING - start with: cd frontend && pnpm dev"
    $criticalCount++
    $allSystemsGo = $false
}

# 1.2 Environment Configuration
Write-Host "`n🔧 Environment Configuration:" -ForegroundColor Yellow
$envPath = "D:\Dev\kha\frontend\.env"
if(Test-Path $envPath) {
    Ok ".env file exists"
    $envContent = Get-Content $envPath -Raw
    
    if($envContent -match "STRIPE_SECRET_KEY=") {
        Ok "Stripe keys configured"
        $successCount++
        
        if($envContent -match "sk_test_") {
            Info "  Using TEST keys (replace for production)"
        } elseif($envContent -match "sk_live_") {
            Ok "  Using LIVE keys - ready for production!"
            $successCount++
        }
    } else {
        Fail "Stripe keys missing"
        $criticalCount++
    }
} else {
    Fail ".env file missing"
    $criticalCount++
}

Header "SECTION 2: EXPORT ASSETS"

# 2.1 GLB Files
Write-Host "`n📦 3D Assets (GLB):" -ForegroundColor Yellow
$glbDir = "D:\Dev\kha\exports\templates"
if(Test-Path $glbDir) {
    $glbFiles = Get-ChildItem "$glbDir\*.glb" -ErrorAction SilentlyContinue
    if($glbFiles) {
        Ok "GLB files: $($glbFiles.Count) found"
        foreach($glb in $glbFiles) {
            Info "  • $($glb.Name) ($('{0:N2}' -f ($glb.Length/1KB)) KB)"
        }
        $successCount++
    } else {
        Warn "No GLB files found"
        $warningCount++
    }
} else {
    Fail "GLB directory missing"
    $criticalCount++
}

# 2.2 KTX2 Textures
Write-Host "`n🎨 Textures (KTX2):" -ForegroundColor Yellow
$ktx2Dir = "D:\Dev\kha\exports\textures_ktx2"
if(Test-Path $ktx2Dir) {
    $ktx2Files = Get-ChildItem "$ktx2Dir\*.ktx2" -ErrorAction SilentlyContinue
    if($ktx2Files) {
        Ok "KTX2 files: $($ktx2Files.Count) found"
        foreach($ktx in $ktx2Files) {
            Info "  • $($ktx.Name) ($('{0:N2}' -f ($ktx.Length/1KB)) KB)"
        }
        $successCount++
    } else {
        Warn "No KTX2 files found"
        $warningCount++
    }
} else {
    Fail "KTX2 directory missing"
    $criticalCount++
}

Header "SECTION 3: WOW PACK PIPELINE"

# 3.1 FFmpeg Installation
Write-Host "`n🎬 Video Processing (FFmpeg):" -ForegroundColor Yellow
$ffmpegPath = "D:\Dev\kha\tools\ffmpeg\ffmpeg.exe"
if(Test-Path $ffmpegPath) {
    Ok "FFmpeg installed"
    
    # Test FFmpeg
    try {
        $version = & $ffmpegPath -version 2>&1 | Select-String "ffmpeg version" | Select-Object -First 1
        if($version) {
            Info "  Version: $version"
            $successCount++
        }
    } catch {
        Warn "  Could not verify FFmpeg version"
    }
} else {
    Fail "FFmpeg NOT installed"
    Info "  Fix: Run .\tools\encode\Install-FFmpeg.ps1"
    $criticalCount++
}

# 3.2 ProRes Masters
Write-Host "`n🎥 ProRes Master Files:" -ForegroundColor Yellow
$mastersReady = $true
$masterFiles = @(
    "holo_flux_loop.mov",
    "mach_lightfield.mov",
    "kinetic_logo_parade.mov"
)

$masterCount = 0
$totalSize = 0
foreach($file in $masterFiles) {
    $path = "D:\Dev\kha\content\wowpack\input\$file"
    if(Test-Path $path) {
        $size = (Get-Item $path).Length
        $totalSize += $size
        Ok "$file ($('{0:N2}' -f ($size/1GB)) GB)"
        $masterCount++
    } else {
        Fail "$file MISSING"
        $mastersReady = $false
    }
}

if($masterCount -eq 3) {
    Ok "All ProRes masters present ($('{0:N2}' -f ($totalSize/1GB)) GB total)"
    $successCount++
} else {
    Fail "Only $masterCount/3 masters present"
    $criticalCount++
}

# 3.3 Encoded Outputs
Write-Host "`n🎞️ Encoded Video Outputs:" -ForegroundColor Yellow
$av1Count = 0
$hdrCount = 0
$sdrCount = 0

# Check AV1
$av1Dir = "D:\Dev\kha\content\wowpack\video\av1"
if(Test-Path $av1Dir) {
    $av1Files = @(Get-ChildItem "$av1Dir\*.mp4" -ErrorAction SilentlyContinue)
    $av1Count = $av1Files.Count
    if($av1Count -gt 0) {
        Ok "AV1 encodes: $av1Count files ready"
        $successCount++
    }
}

# Check HDR/SDR
$hdrDir = "D:\Dev\kha\content\wowpack\video\hdr10"
if(Test-Path $hdrDir) {
    $hdrFiles = @(Get-ChildItem "$hdrDir\*_hdr10.mp4" -ErrorAction SilentlyContinue)
    $sdrFiles = @(Get-ChildItem "$hdrDir\*_sdr.mp4" -ErrorAction SilentlyContinue)
    $hdrCount = $hdrFiles.Count
    $sdrCount = $sdrFiles.Count
    
    if($hdrCount -gt 0) {
        Ok "HDR10 encodes: $hdrCount files ready"
        $successCount++
    }
    if($sdrCount -gt 0) {
        Ok "SDR encodes: $sdrCount files ready"
        $successCount++
    }
}

$totalEncoded = $av1Count + $hdrCount + $sdrCount
if($totalEncoded -gt 0) {
    Ok "Total encoded outputs: $totalEncoded files"
} else {
    Warn "No encoded outputs found - run encoding scripts"
    $warningCount++
}

Header "SECTION 4: INTEGRATION GUIDES"

Write-Host "`n📚 Documentation:" -ForegroundColor Yellow
$snapGuides = @(Get-ChildItem "D:\Dev\kha\docs\guides\snap\*.md" -ErrorAction SilentlyContinue).Count
$tiktokGuides = @(Get-ChildItem "D:\Dev\kha\docs\guides\tiktok\*.md" -ErrorAction SilentlyContinue).Count

if($snapGuides -gt 0) {
    Ok "Snap guides: $snapGuides documents"
    $successCount++
} else {
    Warn "Snap guides missing"
    $warningCount++
}

if($tiktokGuides -gt 0) {
    Ok "TikTok guides: $tiktokGuides documents"
    $successCount++
} else {
    Warn "TikTok guides missing"
    $warningCount++
}

Header "SECTION 5: FINAL READINESS ASSESSMENT"

# Calculate readiness score
$totalChecks = $successCount + $warningCount + $criticalCount
$readinessScore = if($totalChecks -gt 0) { 
    [math]::Round(($successCount / $totalChecks) * 100, 1) 
} else { 0 }

$masterReport.metrics = @{
    success = $successCount
    warnings = $warningCount
    critical = $criticalCount
    total = $totalChecks
    score = $readinessScore
}

# Production gates
$coreReady = ($activePort -ne $null) -and ($criticalCount -eq 0)
$wowPackReady = ($masterCount -eq 3) -and (Test-Path $ffmpegPath)
$outputsReady = $totalEncoded -gt 0

$masterReport.readiness = @{
    core = $coreReady
    wowpack = $wowPackReady
    outputs = $outputsReady
    overall = $coreReady -and $wowPackReady
}

# Save comprehensive report
$reportDir = "D:\Dev\kha\verification_reports"
if(-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
}
$reportFile = Join-Path $reportDir "master_launch_report_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
$masterReport | ConvertTo-Json -Depth 10 | Set-Content $reportFile

Write-Host "`n╔══════════════════════════════════════════════════════════╗" -ForegroundColor Magenta
Write-Host "║                    FINAL VERDICT                          ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Magenta

Write-Host "`n📊 Readiness Metrics:" -ForegroundColor Cyan
Write-Host "  ✅ Successful checks: $successCount" -ForegroundColor Green
Write-Host "  ⚠️  Warnings: $warningCount" -ForegroundColor Yellow
Write-Host "  ❌ Critical issues: $criticalCount" -ForegroundColor Red
Write-Host "  📈 Readiness Score: $readinessScore%" -ForegroundColor Cyan

Write-Host "`n🎯 System Status:" -ForegroundColor Cyan
if($coreReady) {
    Write-Host "  ✅ Core Systems: READY" -ForegroundColor Green
} else {
    Write-Host "  ❌ Core Systems: NOT READY" -ForegroundColor Red
}

if($wowPackReady) {
    Write-Host "  ✅ WOW Pack: READY" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  WOW Pack: PARTIAL" -ForegroundColor Yellow
}

if($outputsReady) {
    Write-Host "  ✅ Video Outputs: ENCODED" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Video Outputs: PENDING" -ForegroundColor Yellow
}

Write-Host "`n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta

if($criticalCount -eq 0 -and $readinessScore -ge 80) {
    Write-Host "`n🚀🚀🚀 SYSTEM IS GO FOR LAUNCH! 🚀🚀🚀" -ForegroundColor Green
    Write-Host "You are $readinessScore% ready for production!" -ForegroundColor Green
    
    if($warningCount -gt 0) {
        Write-Host "`n📝 Minor items to address (non-blocking):" -ForegroundColor Yellow
        if($totalEncoded -eq 0) {
            Write-Host "  • Generate video encodes: cd tools\encode && .\Batch-Encode-Simple.ps1" -ForegroundColor White
        }
    }
    
    Write-Host "`n✨ Demo highlights for tonight:" -ForegroundColor Cyan
    Write-Host "  • /hologram - Live 3D recording with watermark" -ForegroundColor White
    Write-Host "  • /templates - Export GLB/KTX2 pipeline" -ForegroundColor White
    Write-Host "  • /publish - Social publishing readiness" -ForegroundColor White
    if($totalEncoded -gt 0) {
        Write-Host "  • Show $totalEncoded encoded video variants (AV1/HDR/SDR)" -ForegroundColor White
    }
    
} elseif($criticalCount -gt 0) {
    Write-Host "`n⚠️ CRITICAL ISSUES MUST BE FIXED! ⚠️" -ForegroundColor Red
    Write-Host "Address $criticalCount critical issues before launch" -ForegroundColor Red
    
    if(-not $activePort) {
        Write-Host "`nPriority 1: Start dev server" -ForegroundColor Yellow
        Write-Host "  cd D:\Dev\kha\frontend" -ForegroundColor White
        Write-Host "  pnpm dev" -ForegroundColor White
    }
} else {
    Write-Host "`n⚠️ SYSTEM NEEDS ATTENTION ⚠️" -ForegroundColor Yellow
    Write-Host "Readiness at $readinessScore% - some components need work" -ForegroundColor Yellow
}

Write-Host "`n📄 Full report saved: $reportFile" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
Write-Host ""

# Return appropriate exit code
if($criticalCount -eq 0) {
    exit 0
} else {
    exit 1
}