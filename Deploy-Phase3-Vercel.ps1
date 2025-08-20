# PowerShell Deployment Script for Phase 3
# iRis to Vercel Deployment Automation

param(
    [Parameter(Mandatory=$false)]
    [string]$CommitMessage = "iRis launch: recorder + pricing + stripe checkout",
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipTests,
    
    [Parameter(Mandatory=$false)]
    [switch]$TestMode
)

Set-Location "D:\Dev\kha\tori_ui_svelte"
$ErrorActionPreference = "Stop"

Write-Host @"

==================================================
       iRis PHASE 3 - VERCEL DEPLOYMENT
==================================================
"@ -ForegroundColor Cyan

# Step 1: Pre-deployment Checks
Write-Host "`n[1/7] Running Pre-deployment Checks..." -ForegroundColor Yellow

# Check for uncommitted changes
$gitStatus = git status --porcelain
if ($gitStatus) {
    Write-Host "  ⚠ Uncommitted changes detected:" -ForegroundColor Yellow
    $gitStatus | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
} else {
    Write-Host "  ✓ Working directory clean" -ForegroundColor Green
}

# Check current branch
$currentBranch = git branch --show-current
Write-Host "  📍 Current branch: $currentBranch" -ForegroundColor Cyan

# Verify critical files exist
$criticalFiles = @(
    "static/plans.json",
    "src/lib/stores/userPlan.ts",
    "src/lib/components/HologramRecorder.svelte",
    "src/routes/pricing/+page.svelte",
    "src/routes/api/billing/checkout/+server.ts"
)

$allFilesPresent = $true
foreach ($file in $criticalFiles) {
    if (Test-Path $file) {
        Write-Host "  ✓ Found: $file" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Missing: $file" -ForegroundColor Red
        $allFilesPresent = $false
    }
}

if (-not $allFilesPresent) {
    Write-Host "`n❌ Critical files missing. Run Phase 1 setup first." -ForegroundColor Red
    exit 1
}

# Step 2: Install Vercel Adapter
Write-Host "`n[2/7] Installing Vercel Adapter..." -ForegroundColor Yellow

$packageJson = Get-Content "package.json" | ConvertFrom-Json
if ($packageJson.devDependencies.'@sveltejs/adapter-vercel') {
    Write-Host "  ✓ Vercel adapter already installed" -ForegroundColor Green
} else {
    Write-Host "  Installing @sveltejs/adapter-vercel..." -ForegroundColor Cyan
    npm install -D @sveltejs/adapter-vercel
    Write-Host "  ✓ Vercel adapter installed" -ForegroundColor Green
}

# Step 3: Configure svelte.config.js
Write-Host "`n[3/7] Configuring SvelteKit for Vercel..." -ForegroundColor Yellow

if (Test-Path "svelte.config.vercel.js") {
    $useVercelConfig = Read-Host "  Use prepared Vercel config? (y/n)"
    if ($useVercelConfig -eq 'y') {
        Copy-Item "svelte.config.js" "svelte.config.backup.js" -Force
        Copy-Item "svelte.config.vercel.js" "svelte.config.js" -Force
        Write-Host "  ✓ Switched to Vercel adapter configuration" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠ Update svelte.config.js to use @sveltejs/adapter-vercel" -ForegroundColor Yellow
}

# Step 4: Run Tests (optional)
if (-not $SkipTests) {
    Write-Host "`n[4/7] Running Build Test..." -ForegroundColor Yellow
    Write-Host "  Building project..." -ForegroundColor Cyan
    
    try {
        npm run build
        Write-Host "  ✓ Build successful" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Build failed" -ForegroundColor Red
        Write-Host "  Error: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n[4/7] Skipping build test" -ForegroundColor Gray
}

# Step 5: Create vercel.json
Write-Host "`n[5/7] Creating Vercel Configuration..." -ForegroundColor Yellow

$vercelConfig = @{
    buildCommand = "npm run build"
    outputDirectory = ".svelte-kit"
    framework = "sveltekit"
    regions = @("iad1")
}

$vercelConfig | ConvertTo-Json | Out-File "vercel.json" -Encoding UTF8
Write-Host "  ✓ Created vercel.json" -ForegroundColor Green

# Step 6: Git Operations
Write-Host "`n[6/7] Preparing Git Commit..." -ForegroundColor Yellow

# Add files
Write-Host "  Adding files to git..." -ForegroundColor Cyan
git add .

# Show what will be committed
Write-Host "  Files to be committed:" -ForegroundColor Cyan
git status --short | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }

if ($TestMode) {
    Write-Host "`n  🧪 TEST MODE - Skipping actual commit" -ForegroundColor Yellow
} else {
    $proceed = Read-Host "`n  Proceed with commit? (y/n)"
    if ($proceed -eq 'y') {
        git commit -m $CommitMessage
        Write-Host "  ✓ Changes committed" -ForegroundColor Green
        
        $push = Read-Host "  Push to origin/$currentBranch? (y/n)"
        if ($push -eq 'y') {
            git push origin $currentBranch
            Write-Host "  ✓ Pushed to GitHub" -ForegroundColor Green
        }
    }
}

# Step 7: Deployment Instructions
Write-Host "`n[7/7] Vercel Deployment Instructions" -ForegroundColor Yellow

Write-Host @"

==================================================
           MANUAL STEPS FOR VERCEL
==================================================

1. Go to: https://vercel.com
2. Click "Add New Project"
3. Import: Jasonleonardvolk/caviar
4. Configure:
   - Root Directory: tori_ui_svelte
   - Framework: SvelteKit
   
5. Add Environment Variables:

"@ -ForegroundColor White

$envVars = @(
    @{Name="STRIPE_SECRET_KEY"; Value="sk_test_xxx"; Note="(Replace with your key)"}
    @{Name="STRIPE_PRICE_PLUS"; Value="price_xxx"; Note="(Your Plus price ID)"}
    @{Name="STRIPE_PRICE_PRO"; Value="price_xxx"; Note="(Your Pro price ID)"}
    @{Name="STRIPE_SUCCESS_URL"; Value="https://your-app.vercel.app/thank-you"; Note=""}
    @{Name="STRIPE_CANCEL_URL"; Value="https://your-app.vercel.app/pricing?canceled=1"; Note=""}
)

foreach ($var in $envVars) {
    Write-Host "   $($var.Name) = $($var.Value)" -ForegroundColor Cyan
    if ($var.Note) {
        Write-Host "     $($var.Note)" -ForegroundColor Gray
    }
}

Write-Host @"

6. Click "Deploy"
7. Wait for build (~2-3 minutes)
8. Test at your-app.vercel.app

==================================================
          POST-DEPLOYMENT TESTING
==================================================

Test URLs:
- https://your-app.vercel.app/
- https://your-app.vercel.app/hologram-studio
- https://your-app.vercel.app/pricing

Test Flow:
1. Free recording (10s + watermark)
2. Stripe checkout
3. Plus recording (60s, no watermark)

==================================================

"@ -ForegroundColor White

# Create deployment record
$deploymentRecord = @{
    Date = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Branch = $currentBranch
    CommitMessage = $CommitMessage
    Files = $criticalFiles
}

$deploymentRecord | ConvertTo-Json | Out-File "deployment-log.json" -Encoding UTF8
Write-Host "✓ Deployment record saved to deployment-log.json" -ForegroundColor Green

Write-Host "`n🚀 Ready for Vercel deployment!" -ForegroundColor Green
Write-Host "   Follow the manual steps above to complete deployment." -ForegroundColor Cyan