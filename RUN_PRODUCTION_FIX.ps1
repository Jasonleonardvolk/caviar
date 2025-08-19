# TORI/KHA Production Fix Script
# Ensures 100% functionality with all components

Write-Host "`n🚀 TORI/KHA Production Fix - One-Shot Solution" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan

# Check if we're in the right directory
if (-not (Test-Path "enhanced_launcher.py")) {
    Write-Host "❌ Error: Not in TORI/KHA directory!" -ForegroundColor Red
    Write-Host "Please run from: C:\Users\jason\Desktop\tori\kha" -ForegroundColor Yellow
    exit 1
}

# Stage 1: Ensure tools directory exists
Write-Host "`n📁 Stage 1: Setting up tools directory..." -ForegroundColor Green
if (-not (Test-Path "tools")) {
    New-Item -ItemType Directory -Path "tools" | Out-Null
    Write-Host "✅ Created tools directory" -ForegroundColor Green
} else {
    Write-Host "✅ Tools directory exists" -ForegroundColor Green
}

# Stage 2: Check if fix scripts exist in tools
Write-Host "`n🔍 Stage 2: Checking fix scripts..." -ForegroundColor Green

$scripts = @{
    "fix_all_issues_production.py" = "Production-ready comprehensive fix"
    "fix_pydantic_imports.py" = "Pydantic import fixer"
}

$allPresent = $true
foreach ($script in $scripts.Keys) {
    $path = Join-Path "tools" $script
    if (Test-Path $path) {
        Write-Host "✅ Found: $script - $($scripts[$script])" -ForegroundColor Green
    } else {
        Write-Host "❌ Missing: $script" -ForegroundColor Red
        $allPresent = $false
    }
}

if (-not $allPresent) {
    Write-Host "`n⚠️  Some fix scripts are missing!" -ForegroundColor Yellow
    Write-Host "Make sure all scripts are in the tools directory" -ForegroundColor Yellow
    exit 1
}

# Stage 3: Run the production fix
Write-Host "`n🔧 Stage 3: Running production fix..." -ForegroundColor Green
Write-Host "This will:" -ForegroundColor Cyan
Write-Host "  • Fix all Pydantic imports (v2 migration)" -ForegroundColor White
Write-Host "  • Install Penrose similarity engine" -ForegroundColor White
Write-Host "  • Set up Concept Mesh components" -ForegroundColor White
Write-Host "  • Create Soliton API endpoints" -ForegroundColor White
Write-Host "  • Update all configuration files" -ForegroundColor White
Write-Host "  • Ensure 100% functionality" -ForegroundColor White

# Execute the fix
$pythonExe = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonExe) {
    Write-Host "❌ Python not found in PATH!" -ForegroundColor Red
    exit 1
}

Write-Host "`n🚀 Executing fix (this may take a few minutes)..." -ForegroundColor Yellow
& python tools\fix_all_issues_production.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Fix completed successfully!" -ForegroundColor Green
    
    # Stage 4: Offer to start the launcher
    Write-Host "`n🎯 Ready to start TORI!" -ForegroundColor Cyan
    Write-Host "Would you like to start the enhanced launcher now? (Y/N)" -ForegroundColor Yellow
    $response = Read-Host
    
    if ($response -eq 'Y' -or $response -eq 'y') {
        Write-Host "`n🚀 Starting enhanced launcher..." -ForegroundColor Green
        Start-Process python -ArgumentList "enhanced_launcher.py" -NoNewWindow
    } else {
        Write-Host "`n📋 To start manually, run:" -ForegroundColor Cyan
        Write-Host "    python enhanced_launcher.py" -ForegroundColor White
    }
    
    # Provide test command
    Write-Host "`n🧪 To test all components after startup:" -ForegroundColor Cyan
    Write-Host "    python test_components.py" -ForegroundColor White
} else {
    Write-Host "`n❌ Fix failed! Check the output above for errors." -ForegroundColor Red
    exit 1
}

Write-Host "`n🎉 Done!" -ForegroundColor Green
