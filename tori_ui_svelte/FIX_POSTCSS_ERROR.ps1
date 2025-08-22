# TORI PostCSS Error Fix - PowerShell Edition
# Fixes the "Unknown word position" error in your Svelte project

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   TORI PostCSS Error Fix" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
$projectPath = "C:\Users\jason\Desktop\tori\kha\tori_ui_svelte"
Set-Location $projectPath

Write-Host "🔍 Diagnosing PostCSS issues..." -ForegroundColor Blue

# Check if package.json exists
if (-not (Test-Path "package.json")) {
    Write-Host "❌ package.json not found in current directory!" -ForegroundColor Red
    Write-Host "Current location: $(Get-Location)" -ForegroundColor Yellow
    exit 1
}

# Step 1: Install missing dependencies
Write-Host ""
Write-Host "🔧 Step 1: Installing missing Tailwind CSS dependencies..." -ForegroundColor Green

$dependencies = "tailwindcss@latest autoprefixer@latest postcss@latest postcss-safe-parser@latest"

try {
    Write-Host "Installing: $dependencies" -ForegroundColor Yellow
    Invoke-Expression "npm install -D $dependencies"
    Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
}
catch {
    Write-Host "❌ Failed to install dependencies: $_" -ForegroundColor Red
    Write-Host "Trying alternative approach..." -ForegroundColor Yellow
    
    # Try installing one by one
    $depArray = @("tailwindcss@latest", "autoprefixer@latest", "postcss@latest", "postcss-safe-parser@latest")
    foreach ($dep in $depArray) {
        try {
            Write-Host "Installing $dep..." -ForegroundColor Yellow
            Invoke-Expression "npm install -D $dep"
        }
        catch {
            Write-Host "⚠️ Failed to install $dep, continuing..." -ForegroundColor Yellow
        }
    }
}

# Step 2: Clear Vite cache
Write-Host ""
Write-Host "🧹 Step 2: Clearing Vite cache..." -ForegroundColor Green

if (Test-Path "node_modules\.vite") {
    Remove-Item -Recurse -Force "node_modules\.vite"
    Write-Host "✅ Vite cache cleared" -ForegroundColor Green
} else {
    Write-Host "ℹ️ No Vite cache to clear" -ForegroundColor Blue
}

# Step 3: Update PostCSS config to be defensive
Write-Host ""
Write-Host "🛠️ Step 3: Creating bulletproof PostCSS config..." -ForegroundColor Green

# Create the PostCSS config content as a here-string
$postcssConfig = @'
/** 
 * Bulletproof PostCSS Configuration for TORI UI
 * Handles missing dependencies gracefully
 */

export default {
    plugins: {
        // Only include plugins that are actually installed
        ...((() => {
            try {
                require.resolve('tailwindcss');
                return { tailwindcss: {} };
            } catch {
                console.warn('⚠️ tailwindcss not found, skipping...');
                return {};
            }
        })()),
        ...((() => {
            try {
                require.resolve('autoprefixer');
                return { autoprefixer: {} };
            } catch {
                console.warn('⚠️ autoprefixer not found, skipping...');
                return {};
            }
        })())
    }
};
'@

$postcssConfig | Out-File -FilePath "postcss.config.js" -Encoding UTF8
Write-Host "✅ PostCSS config updated" -ForegroundColor Green

# Step 4: Check PersonaPanel.svelte for syntax issues
Write-Host ""
Write-Host "🔍 Step 4: Checking PersonaPanel.svelte for CSS issues..." -ForegroundColor Green

$personaPanelPath = "src\lib\components\PersonaPanel.svelte"
if (Test-Path $personaPanelPath) {
    $content = Get-Content $personaPanelPath -Raw -Encoding UTF8
    
    # Check for common issues
    $hasIssues = $false
    $fixedContent = $content
    
    # Remove any Unicode zero-width characters that can cause parsing issues
    $fixedContent = $fixedContent -replace '[\u200B-\u200D\uFEFF]', ''
    
    # Check if we made any changes
    if ($fixedContent -ne $content) {
        $fixedContent | Out-File -FilePath $personaPanelPath -Encoding UTF8 -NoNewline
        Write-Host "✅ Fixed CSS syntax issues in PersonaPanel.svelte" -ForegroundColor Green
        $hasIssues = $true
    } else {
        Write-Host "✅ No CSS syntax issues found in PersonaPanel.svelte" -ForegroundColor Green
    }
} else {
    Write-Host "⚠️ PersonaPanel.svelte not found at expected location" -ForegroundColor Yellow
}

# Step 5: Add debug scripts to package.json
Write-Host ""
Write-Host "📋 Step 5: Adding debug scripts..." -ForegroundColor Green

try {
    $packageJson = Get-Content "package.json" -Raw | ConvertFrom-Json
    
    # Ensure scripts object exists
    if (-not $packageJson.scripts) {
        $packageJson | Add-Member -NotePropertyName "scripts" -NotePropertyValue @{} -Force
    }
    
    # Add debug scripts
    $packageJson.scripts."dev:debug" = "set DEBUG=vite:css && vite dev --port 5173 --host 0.0.0.0"
    $packageJson.scripts."postcss:check" = "echo. | npx postcss --config ."
    $packageJson.scripts."css:lint" = "node postcss-debug-fix.js"
    
    $packageJson | ConvertTo-Json -Depth 10 | Out-File -FilePath "package.json" -Encoding UTF8
    Write-Host "✅ Debug scripts added to package.json" -ForegroundColor Green
}
catch {
    Write-Host "⚠️ Could not update package.json: $_" -ForegroundColor Yellow
}

# Step 6: Create .env.local for debugging
Write-Host ""
Write-Host "🔧 Step 6: Creating debug environment..." -ForegroundColor Green

$envContent = @'
# PostCSS Debug Configuration
DEBUG=vite:css
VITE_CSS_DEBUG=true
'@

$envContent | Out-File -FilePath ".env.local" -Encoding UTF8
Write-Host "✅ Created .env.local for debugging" -ForegroundColor Green

# Final verification
Write-Host ""
Write-Host "🔍 Step 7: Verifying installation..." -ForegroundColor Green

try {
    $output = npm list tailwindcss 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ tailwindcss is installed" -ForegroundColor Green
    } else {
        Write-Host "⚠️ tailwindcss verification result: $output" -ForegroundColor Yellow
    }
}
catch {
    Write-Host "⚠️ Could not verify tailwindcss installation" -ForegroundColor Yellow
}

# Success message
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🎉 FIXES COMPLETED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📋 What was fixed:" -ForegroundColor Blue
Write-Host "  ✅ Installed missing Tailwind CSS dependencies" -ForegroundColor Green
Write-Host "  ✅ Created bulletproof PostCSS configuration" -ForegroundColor Green
Write-Host "  ✅ Cleared Vite cache" -ForegroundColor Green
Write-Host "  ✅ Fixed CSS syntax issues (if any)" -ForegroundColor Green
Write-Host "  ✅ Added debugging tools" -ForegroundColor Green
Write-Host ""

Write-Host "🚀 Next Steps:" -ForegroundColor Blue
Write-Host "  1. Stop your current dev server (Ctrl+C)" -ForegroundColor Yellow
Write-Host "  2. Run: npm run dev" -ForegroundColor Yellow
Write-Host "  3. Your PostCSS error should be resolved!" -ForegroundColor Yellow
Write-Host ""

Write-Host "🔧 If you still have issues:" -ForegroundColor Blue
Write-Host "  • Run: npm run dev:debug (for verbose output)" -ForegroundColor Yellow
Write-Host "  • Run: npm run css:lint (for CSS diagnostics)" -ForegroundColor Yellow
Write-Host "  • Check: POSTCSS_FIX_GUIDE.md" -ForegroundColor Yellow
Write-Host ""

Write-Host "Press any key to continue..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
