# PowerShell Script for Phase 2 Local Testing
# iRis Hologram Studio - Automated Test Runner

param(
    [switch]$SkipInstall,
    [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"
Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host @"

==================================================
    iRis PHASE 2 - LOCAL TEST CHECKLIST
==================================================
"@ -ForegroundColor Cyan

# Step 1: Install Dependencies
if (!$SkipInstall) {
    Write-Host "`n[STEP 1] Installing Dependencies..." -ForegroundColor Yellow
    Write-Host "  Running: pnpm install" -ForegroundColor Gray
    
    try {
        pnpm install
        Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Failed to install dependencies" -ForegroundColor Red
        Write-Host "  Trying npm instead..." -ForegroundColor Yellow
        npm install
    }
} else {
    Write-Host "`n[STEP 1] Skipping dependency installation" -ForegroundColor Gray
}

# Step 2: Start Dev Server
Write-Host "`n[STEP 2] Starting Development Server..." -ForegroundColor Yellow

# Kill existing processes on port 5173
$existingProcess = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
if ($existingProcess) {
    Write-Host "  Stopping existing process on port 5173..." -ForegroundColor Gray
    Stop-Process -Id $existingProcess.OwningProcess -Force
    Start-Sleep -Seconds 2
}

Write-Host "  Running: pnpm dev" -ForegroundColor Gray
$devServer = Start-Process "cmd" -ArgumentList "/c pnpm dev" -PassThru -WindowStyle Minimized

# Wait for server to be ready
Write-Host "  Waiting for server..." -ForegroundColor Gray
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {}
    Start-Sleep -Seconds 1
    Write-Host "." -NoNewline
}
Write-Host ""

if ($ready) {
    Write-Host "  ✓ Server running at http://localhost:5173" -ForegroundColor Green
} else {
    Write-Host "  ✗ Server failed to start" -ForegroundColor Red
    exit 1
}

# Step 3: Open Test Pages
Write-Host "`n[STEP 3] Opening Test Pages..." -ForegroundColor Yellow
Start-Process "http://localhost:5173/hologram"
Start-Sleep -Seconds 2
Write-Host "  ✓ Opened: /hologram" -ForegroundColor Green

# Step 4: Test Free Plan
Write-Host "`n[TEST 1] FREE PLAN RECORDING" -ForegroundColor Magenta
Write-Host @"
  
  Manual Steps:
  1. Click 'Record 10s' button
  2. Watch countdown (10 → 0)
  3. File downloads as iris_*.webm
  4. Open file and verify watermark in lower-right:
     - "iRis • hologram studio"
     - "Created with CAVIAR"
"@ -ForegroundColor White

Write-Host "`n  Press ENTER when test complete..." -ForegroundColor Yellow
Read-Host

$freeTestPassed = Read-Host "  Did watermark appear correctly? (y/n)"
if ($freeTestPassed -eq 'y') {
    Write-Host "  ✓ Free plan test PASSED" -ForegroundColor Green
} else {
    Write-Host "  ✗ Free plan test FAILED" -ForegroundColor Red
}

# Step 5: Stripe Configuration Check
Write-Host "`n[TEST 2] STRIPE CONFIGURATION" -ForegroundColor Magenta

# Check .env file
$envFile = Get-Content ".env" -Raw
$hasStripeKey = $envFile -match "STRIPE_SECRET_KEY=sk_test_"
$hasPlusPrice = $envFile -match "STRIPE_PRICE_PLUS=price_"
$hasProPrice = $envFile -match "STRIPE_PRICE_PRO=price_"

if ($hasStripeKey -and $hasPlusPrice -and $hasProPrice) {
    Write-Host "  ✓ Stripe configuration found in .env" -ForegroundColor Green
} else {
    Write-Host @"
  
  ⚠ Stripe not configured. Please:
  1. Go to https://dashboard.stripe.com
  2. Create 'iRis Plus' product (monthly)
  3. Create 'iRis Pro' product (monthly)
  4. Update .env with Price IDs:
     STRIPE_SECRET_KEY=sk_test_...
     STRIPE_PRICE_PLUS=price_...
     STRIPE_PRICE_PRO=price_...
"@ -ForegroundColor Yellow
    
    Write-Host "`n  Press ENTER after updating .env..." -ForegroundColor Yellow
    Read-Host
    
    # Restart server to load new env
    Write-Host "  Restarting server with new config..." -ForegroundColor Gray
    Stop-Process -Id $devServer.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    $devServer = Start-Process "cmd" -ArgumentList "/c pnpm dev" -PassThru -WindowStyle Minimized
    Start-Sleep -Seconds 5
}

# Open pricing page
Start-Process "http://localhost:5173/pricing"
Write-Host "  ✓ Opened: /pricing" -ForegroundColor Green

Write-Host @"

  Manual Steps:
  1. Click 'Get Plus' button
  2. Stripe Checkout should open
  3. Enter test card: 4242 4242 4242 4242
  4. Expiry: 12/34, CVC: 123
  5. Complete payment
  6. Should redirect to /thank-you?plan=plus
"@ -ForegroundColor White

Write-Host "`n  Press ENTER when test complete..." -ForegroundColor Yellow
Read-Host

$stripeTestPassed = Read-Host "  Did checkout complete successfully? (y/n)"
if ($stripeTestPassed -eq 'y') {
    Write-Host "  ✓ Stripe checkout test PASSED" -ForegroundColor Green
} else {
    Write-Host "  ✗ Stripe checkout test FAILED" -ForegroundColor Red
}

# Step 6: Test Plus Plan
Write-Host "`n[TEST 3] PLUS PLAN RECORDING" -ForegroundColor Magenta

# Return to hologram page
Start-Process "http://localhost:5173/hologram"
Write-Host "  ✓ Returned to: /hologram" -ForegroundColor Green

Write-Host @"

  Manual Steps:
  1. Verify 'Plus' pill shows in recorder
  2. Click 'Record 60s' button
  3. Record for a few seconds
  4. Stop recording
  5. Open downloaded file
  6. Verify NO watermark present
"@ -ForegroundColor White

Write-Host "`n  Press ENTER when test complete..." -ForegroundColor Yellow
Read-Host

$plusTestPassed = Read-Host "  Was watermark absent (no watermark)? (y/n)"
if ($plusTestPassed -eq 'y') {
    Write-Host "  ✓ Plus plan test PASSED" -ForegroundColor Green
} else {
    Write-Host "  ✗ Plus plan test FAILED" -ForegroundColor Red
}

# Step 7: Optional Webhook Test
Write-Host "`n[TEST 4] WEBHOOK (Optional)" -ForegroundColor Magenta
$testWebhook = Read-Host "  Test webhook? (y/n)"

if ($testWebhook -eq 'y') {
    Write-Host @"

  Manual Steps:
  1. Install Stripe CLI: https://stripe.com/docs/stripe-cli
  2. Open new terminal
  3. Run: stripe login
  4. Run: stripe listen --forward-to localhost:5173/api/billing/webhook
  5. Repeat a checkout
  6. Verify webhook event in CLI output
"@ -ForegroundColor White
    
    Write-Host "`n  Press ENTER when test complete..." -ForegroundColor Yellow
    Read-Host
}

# Test Summary
Write-Host "`n===================================================" -ForegroundColor Cyan
Write-Host "              TEST SUMMARY" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

$results = @(
    @{Name="Free Plan (watermark)"; Passed=($freeTestPassed -eq 'y')}
    @{Name="Stripe Checkout"; Passed=($stripeTestPassed -eq 'y')}
    @{Name="Plus Plan (no watermark)"; Passed=($plusTestPassed -eq 'y')}
)

$allPassed = $true
foreach ($result in $results) {
    $icon = if ($result.Passed) { "✓" } else { "✗"; $allPassed = $false }
    $color = if ($result.Passed) { "Green" } else { "Red" }
    Write-Host "  $icon $($result.Name)" -ForegroundColor $color
}

if ($allPassed) {
    Write-Host "`n🎉 ALL TESTS PASSED! Ready for Phase 3" -ForegroundColor Green
} else {
    Write-Host "`n⚠ Some tests failed. Please review and fix." -ForegroundColor Yellow
}

# Cleanup
if (!$KeepRunning) {
    Write-Host "`nStopping dev server..." -ForegroundColor Gray
    Stop-Process -Id $devServer.Id -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "`nDev server still running at http://localhost:5173" -ForegroundColor Gray
}

Write-Host "`nPhase 2 Testing Complete!" -ForegroundColor Cyan