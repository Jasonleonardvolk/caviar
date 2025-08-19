param([switch]$EncodeVideos = $false)

Write-Host "`n" -NoNewline
Write-Host "🚀 CAVIAR LAUNCH READINESS CHECK 🚀" -ForegroundColor Magenta
Write-Host "=====================================" -ForegroundColor Magenta
Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
Write-Host ""

function Ok($m){Write-Host "[✅] $m" -f Green}
function Info($m){Write-Host "[📌] $m" -f Cyan}
function Warn($m){Write-Host "[⚠️] $m" -f Yellow}
function Fail($m){Write-Host "[❌] $m" -f Red}

$allGood = $true

# 1. DEV SERVER CHECK
Write-Host "1️⃣  DEV SERVER STATUS" -ForegroundColor Yellow
Write-Host "---------------------" -ForegroundColor DarkGray

function TestPort($port) { 
    try { 
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health/ping" -TimeoutSec 2 -UseBasicParsing
        return $response.StatusCode -eq 200
    } catch { 
        return $false 
    } 
}

$port = $null
foreach($p in @(5173, 3000, 3310)) { 
    if (TestPort $p) { 
        $port = $p
        break 
    } 
}

if ($port) {
    Ok "Dev server running on port $port"
    
    # Test key routes
    $routes = @("/", "/hologram", "/templates", "/pricing")
    foreach ($route in $routes) {
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$port$route" -TimeoutSec 3 -UseBasicParsing
            if ($resp.StatusCode -eq 200) {
                Ok "  $route → 200 OK"
            }
        } catch {
            Fail "  $route → Failed"
            $allGood = $false
        }
    }
} else {
    Fail "No dev server running!"
    Info "  Start with: cd frontend && pnpm.cmd dev"
    $allGood = $false
}

# 2. PRODUCTION FILES
Write-Host "`n2️⃣  PRODUCTION ASSETS" -ForegroundColor Yellow
Write-Host "--------------------" -ForegroundColor DarkGray

$assets = @{
    ".env with Stripe" = "D:\Dev\kha\frontend\.env"
    "Concept Graph" = "D:\Dev\kha\data\concept_graph.json"
    "GLB Template" = "D:\Dev\kha\exports\templates\sample_hologram.glb"
    "KTX2 Texture" = "D:\Dev\kha\exports\textures_ktx2\sample_texture.ktx2"
    "Snap Guide" = "D:\Dev\kha\integrations\snap\guides\snap_integration_guide.md"
    "TikTok Guide" = "D:\Dev\kha\integrations\tiktok\guides\tiktok_integration_guide.md"
}

foreach ($asset in $assets.GetEnumerator()) {
    if (Test-Path $asset.Value) {
        Ok "$($asset.Key)"
    } else {
        Fail "$($asset.Key) - Missing!"
        $allGood = $false
    }
}

# 3. WOW PACK STATUS
Write-Host "`n3️⃣  WOW PACK MEDIA" -ForegroundColor Yellow
Write-Host "-----------------" -ForegroundColor DarkGray

# Check FFmpeg
$ffmpeg = "D:\Dev\kha\tools\ffmpeg\ffmpeg.exe"
if (Test-Path $ffmpeg) {
    Ok "FFmpeg installed"
    # Add to PATH silently
    if ($env:Path -notlike "*D:\Dev\kha\tools\ffmpeg*") {
        $env:Path = "D:\Dev\kha\tools\ffmpeg;$env:Path"
    }
} else {
    Fail "FFmpeg missing - run tools\encode\Install-FFmpeg.ps1"
    $allGood = $false
}

# Check ProRes masters
$masters = @(
    "holo_flux_loop.mov",
    "mach_lightfield.mov",
    "kinetic_logo_parade.mov"
)

$masterCount = 0
foreach ($master in $masters) {
    $path = "D:\Dev\kha\content\wowpack\input\$master"
    if (Test-Path $path) {
        $size = [math]::Round((Get-Item $path).Length / 1MB, 1)
        Ok "$master ($size MB)"
        $masterCount++
    } else {
        Fail "$master missing"
    }
}

if ($masterCount -eq 3) {
    Info "All ProRes masters present!"
    
    if ($EncodeVideos) {
        Write-Host "`n  🎬 Encoding videos..." -ForegroundColor Cyan
        Push-Location "D:\Dev\kha\tools\encode"
        & .\Batch-Encode-Simple.ps1
        Pop-Location
        Ok "Videos encoded!"
    } else {
        Info "  Run with -EncodeVideos to generate H264/HEVC/AV1"
    }
}

# 4. HEALTH ENDPOINT
Write-Host "`n4️⃣  SYSTEM HEALTH" -ForegroundColor Yellow
Write-Host "---------------" -ForegroundColor DarkGray

if ($port) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 5
        
        if ($health.readiness.demo -eq $true) {
            Ok "System reports DEMO READY ✨"
        } else {
            Warn "System not fully ready for demo"
        }
        
        # Show key metrics
        Info "  Stripe configured: $($health.monetization.stripeKeyPresent)"
        Info "  Templates ready: GLB=$($health.templates.counts.glb) KTX2=$($health.templates.counts.ktx2)"
        Info "  Guides: Snap=$($health.guides.snap) TikTok=$($health.guides.tiktok)"
        
    } catch {
        Fail "Health check failed"
        $allGood = $false
    }
}

# 5. FINAL STATUS
Write-Host "`n=====================================" -ForegroundColor Magenta

if ($allGood -and $masterCount -eq 3) {
    Write-Host "✅ 100% READY FOR LAUNCH TONIGHT! 🚀" -ForegroundColor Green -BackgroundColor DarkGreen
    Write-Host ""
    Write-Host "📋 Quick Links:" -ForegroundColor Cyan
    Write-Host "  Hologram Demo:  " -NoNewline -f White
    Write-Host "http://localhost:$port/hologram" -f Blue
    Write-Host "  Templates:      " -NoNewline -f White
    Write-Host "http://localhost:$port/templates" -f Blue
    Write-Host "  Pricing:        " -NoNewline -f White
    Write-Host "http://localhost:$port/pricing" -f Blue
    Write-Host ""
    Write-Host "💎 Pro Tips for Tonight:" -ForegroundColor Magenta
    Write-Host "  • Show the hologram recorder capturing live content" -f White
    Write-Host "  • Export a concept graph to GLB from /templates" -f White
    Write-Host "  • Display the WOW Pack videos as 'social-ready masters'" -f White
    Write-Host "  • Mention Snap/TikTok AR integration capabilities" -f White
    
} elseif ($allGood) {
    Write-Host "✅ CORE SYSTEM READY (Video encoding optional)" -ForegroundColor Green
    Write-Host "  Run with -EncodeVideos flag to generate video assets" -f Yellow
} else {
    Write-Host "⚠️  SOME ISSUES NEED FIXING" -ForegroundColor Yellow -BackgroundColor DarkRed
    Write-Host "  Review the failures above and fix before launch" -f White
}

Write-Host ""
Write-Host "🎯 GO CRUSH IT TONIGHT! 🎯" -ForegroundColor Magenta
Write-Host ""