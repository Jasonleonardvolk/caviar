# PowerShell Test Script for Phase 2
# Run this to automate Phase 2 testing

Write-Host "=== iRis Phase 2 Test Suite ===" -ForegroundColor Cyan

# Configuration
$baseUrl = "http://localhost:5173"
$projectPath = "D:\Dev\kha\tori_ui_svelte"

# Step 1: Check Prerequisites
Write-Host "`n[1/7] Checking Prerequisites..." -ForegroundColor Yellow

# Check if files exist
$requiredFiles = @(
    "$projectPath\static\plans.json",
    "$projectPath\src\lib\stores\userPlan.ts",
    "$projectPath\src\lib\components\HologramRecorder.svelte",
    "$projectPath\src\routes\pricing\+page.svelte",
    "$projectPath\src\routes\thank-you\+page.svelte"
)

$allFilesExist = $true
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ Found: $(Split-Path $file -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Missing: $(Split-Path $file -Leaf)" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if (-not $allFilesExist) {
    Write-Host "`nERROR: Missing required files. Run Phase 1 first." -ForegroundColor Red
    exit 1
}

# Step 2: Check .env configuration
Write-Host "`n[2/7] Checking .env Configuration..." -ForegroundColor Yellow

$envPath = "$projectPath\.env"
if (Test-Path $envPath) {
    $envContent = Get-Content $envPath -Raw
    
    $stripeKeys = @(
        "STRIPE_SECRET_KEY",
        "STRIPE_PRICE_PLUS", 
        "STRIPE_PRICE_PRO"
    )
    
    foreach ($key in $stripeKeys) {
        if ($envContent -match "$key=(.+)") {
            $value = $matches[1]
            if ($value -eq "sk_test_xxx" -or $value -eq "price_PLUS_TEST" -or $value -eq "price_PRO_TEST") {
                Write-Host "  ⚠ $key needs real test value (currently: $value)" -ForegroundColor Yellow
            } else {
                Write-Host "  ✓ $key configured" -ForegroundColor Green
            }
        } else {
            Write-Host "  ✗ $key missing" -ForegroundColor Red
        }
    }
} else {
    Write-Host "  ✗ .env file missing" -ForegroundColor Red
}

# Step 3: Install Dependencies
Write-Host "`n[3/7] Installing Dependencies..." -ForegroundColor Yellow
Set-Location $projectPath

if (Get-Command pnpm -ErrorAction SilentlyContinue) {
    Write-Host "  Installing with pnpm..." -ForegroundColor Cyan
    pnpm install
} else {
    Write-Host "  pnpm not found, using npm..." -ForegroundColor Cyan
    npm install
}

# Check if stripe is installed
$packageJson = Get-Content "package.json" | ConvertFrom-Json
if ($packageJson.dependencies.stripe) {
    Write-Host "  ✓ Stripe SDK installed" -ForegroundColor Green
} else {
    Write-Host "  Installing Stripe SDK..." -ForegroundColor Cyan
    npm install stripe
}

# Step 4: Start Dev Server
Write-Host "`n[4/7] Starting Dev Server..." -ForegroundColor Yellow
Write-Host "  Starting server on $baseUrl" -ForegroundColor Cyan

# Kill any existing process on port 5173
$existingProcess = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
if ($existingProcess) {
    Write-Host "  Killing existing process on port 5173..." -ForegroundColor Yellow
    Stop-Process -Id $existingProcess.OwningProcess -Force
    Start-Sleep -Seconds 2
}

# Start dev server in background
$devProcess = Start-Process -FilePath "cmd" -ArgumentList "/c", "pnpm dev" -WorkingDirectory $projectPath -PassThru -WindowStyle Hidden

# Wait for server to start
Write-Host "  Waiting for server to start..." -ForegroundColor Cyan
$attempts = 0
$maxAttempts = 30
while ($attempts -lt $maxAttempts) {
    try {
        $response = Invoke-WebRequest -Uri $baseUrl -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "  ✓ Server running at $baseUrl" -ForegroundColor Green
            break
        }
    } catch {
        # Server not ready yet
    }
    $attempts++
    Start-Sleep -Seconds 1
}

if ($attempts -eq $maxAttempts) {
    Write-Host "  ✗ Server failed to start" -ForegroundColor Red
    exit 1
}

# Step 5: Open Browser for Manual Testing
Write-Host "`n[5/7] Opening Browser for Manual Testing..." -ForegroundColor Yellow

# Create test URLs
$testUrls = @{
    "Hologram Studio" = "$baseUrl/hologram-studio"
    "Pricing Page" = "$baseUrl/pricing"
    "Regular Hologram" = "$baseUrl/hologram"
}

Write-Host "`n  Test URLs:" -ForegroundColor Cyan
foreach ($page in $testUrls.Keys) {
    Write-Host "    $page`: $($testUrls[$page])" -ForegroundColor White
}

# Open hologram studio in default browser
Start-Process "$baseUrl/hologram-studio"

# Step 6: Test Checklist
Write-Host "`n[6/7] Manual Test Checklist:" -ForegroundColor Yellow
Write-Host @"

  FREE PLAN TESTING:
  ----------------
  [ ] Navigate to /hologram-studio
  [ ] Verify "Free" pill shows in recorder bar
  [ ] Click "Record 10s" button
  [ ] Verify countdown from 10 to 0
  [ ] Verify file downloads as iris_*.webm
  [ ] Open video and verify watermark present

  STRIPE TESTING:
  -------------
  [ ] Navigate to /pricing
  [ ] Click "Get Plus" button
  [ ] Enter test card: 4242 4242 4242 4242
  [ ] Complete checkout
  [ ] Verify redirect to /thank-you

  PLUS PLAN TESTING:
  ----------------
  [ ] Return to /hologram-studio
  [ ] Verify "Plus" pill shows
  [ ] Verify "Record 60s" button
  [ ] Record video
  [ ] Verify NO watermark in video

"@ -ForegroundColor White

# Step 7: Cleanup Options
Write-Host "[7/7] Test Complete!" -ForegroundColor Green
Write-Host "`nOptions:" -ForegroundColor Yellow
Write-Host "  [K] Keep server running" -ForegroundColor Cyan
Write-Host "  [S] Stop server" -ForegroundColor Cyan
Write-Host "  [R] Reset to Free plan" -ForegroundColor Cyan
Write-Host "  [Q] Quit" -ForegroundColor Cyan

$choice = Read-Host "`nSelect option"

switch ($choice.ToUpper()) {
    "K" {
        Write-Host "Server still running at $baseUrl" -ForegroundColor Green
    }
    "S" {
        if ($devProcess -and -not $devProcess.HasExited) {
            Stop-Process -Id $devProcess.Id -Force
            Write-Host "Server stopped" -ForegroundColor Yellow
        }
    }
    "R" {
        # Reset localStorage via JavaScript
        $resetScript = @"
localStorage.removeItem('iris.plan');
console.log('Plan reset to Free');
"@
        Write-Host "To reset plan, run this in browser console:" -ForegroundColor Yellow
        Write-Host $resetScript -ForegroundColor Cyan
    }
    default {
        if ($devProcess -and -not $devProcess.HasExited) {
            Stop-Process -Id $devProcess.Id -Force
        }
    }
}

Write-Host "`n=== Phase 2 Testing Complete ===" -ForegroundColor Green