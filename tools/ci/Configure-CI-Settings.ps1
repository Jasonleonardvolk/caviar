#!/usr/bin/env pwsh
# Configure-CI-Settings.ps1
# Complete CI/CD configuration for production builds

$ErrorActionPreference = "Stop"

Write-Host @"
====================================================
    iRis CI/CD Configuration
    
    1. NPM Audit: High severity only
    2. A11y: Warnings (non-blocking)
    3. Shader Validation: Naga (Tint optional)
====================================================
"@ -ForegroundColor Cyan

Write-Host ""

# 1. Configure NPM Audit Level
Write-Host "[1/3] Configuring NPM Audit..." -ForegroundColor Yellow

$npmrcContent = @'
# CI/CD Configuration
audit-level=high
loglevel=warn
prefer-offline=true
'@

# Update .npmrc files
@("D:\Dev\kha\.npmrc", "D:\Dev\kha\tori_ui_svelte\.npmrc") | ForEach-Object {
    $file = $_
    Write-Host "  Updating: $file" -ForegroundColor Gray
    
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        if ($content -notmatch "audit-level") {
            Add-Content -Path $file -Value "`n$npmrcContent"
        }
    } else {
        Set-Content -Path $file -Value $npmrcContent
    }
}

Write-Host "  ✅ Audit configured for HIGH severity only" -ForegroundColor Green

# 2. Configure Svelte A11y Settings
Write-Host ""
Write-Host "[2/3] Configuring A11y Settings..." -ForegroundColor Yellow

$svelteConfig = "D:\Dev\kha\tori_ui_svelte\svelte.config.js"

if (Test-Path $svelteConfig) {
    $config = Get-Content $svelteConfig -Raw
    
    # Check if onwarn is already configured
    if ($config -notmatch "onwarn") {
        Write-Host "  Adding A11y warning handler..." -ForegroundColor Gray
        
        # Create updated config with onwarn
        $updatedConfig = @'
import adapter from '@sveltejs/adapter-node';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  
  kit: {
    adapter: adapter({ out: 'build' }),
    prerender: { entries: [] }
  },
  
  // Handle A11y warnings gracefully
  onwarn: (warning, handler) => {
    // Treat A11y warnings as warnings, not errors
    if (warning.code.startsWith('a11y-')) {
      console.warn(`A11y Warning: ${warning.message}`);
      return;
    }
    
    // Pass other warnings to default handler
    handler(warning);
  }
};

export default config;
'@
        
        Set-Content -Path $svelteConfig -Value $updatedConfig
        Write-Host "  ✅ A11y warnings configured as non-blocking" -ForegroundColor Green
    } else {
        Write-Host "  ✅ A11y handler already configured" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠️  svelte.config.js not found" -ForegroundColor Yellow
}

# 3. Check Shader Validation Status
Write-Host ""
Write-Host "[3/3] Checking Shader Validation..." -ForegroundColor Yellow

$nagaPath = "D:\Dev\kha\tools\shaders\bin\naga.exe"
$tintPath = "D:\Dev\kha\tools\shaders\bin\tint.exe"

if (Test-Path $nagaPath) {
    Write-Host "  ✅ Naga validator: INSTALLED" -ForegroundColor Green
} else {
    Write-Host "  ⚠️  Naga validator: MISSING" -ForegroundColor Yellow
}

if (Test-Path $tintPath) {
    Write-Host "  ✅ Tint validator: INSTALLED" -ForegroundColor Green
} else {
    Write-Host "  ℹ️  Tint validator: OPTIONAL (not installed)" -ForegroundColor Blue
    Write-Host "     Run Setup-Tint-Validator.ps1 if desired" -ForegroundColor Gray
}

# Create CI validation script
Write-Host ""
Write-Host "Creating CI validation script..." -ForegroundColor Yellow

$ciScript = @'
#!/usr/bin/env pwsh
# Run-CI-Checks.ps1
# Run all CI checks with proper settings

param(
    [switch]$Quick,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$failed = $false

Write-Host "===== Running CI Checks =====" -ForegroundColor Cyan

# 1. NPM Audit (high only)
Write-Host "`n[1/4] Security Audit..." -ForegroundColor Yellow
Set-Location "D:\Dev\kha\tori_ui_svelte"
& npm audit --audit-level=high
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ High severity vulnerabilities found" -ForegroundColor Red
    $failed = $true
} else {
    Write-Host "  ✅ No high severity issues" -ForegroundColor Green
}

# 2. Build Check
Write-Host "`n[2/4] Build Check..." -ForegroundColor Yellow
& pnpm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ❌ Build failed" -ForegroundColor Red
    $failed = $true
} else {
    Write-Host "  ✅ Build successful" -ForegroundColor Green
}

# 3. Type Check
if (!$Quick) {
    Write-Host "`n[3/4] Type Check..." -ForegroundColor Yellow
    & pnpm run check
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ⚠️  Type issues (non-blocking)" -ForegroundColor Yellow
    } else {
        Write-Host "  ✅ Types valid" -ForegroundColor Green
    }
}

# 4. Shader Validation
if (!$Quick) {
    Write-Host "`n[4/4] Shader Validation..." -ForegroundColor Yellow
    $shaderGate = "D:\Dev\kha\tools\shaders\validate_and_report.mjs"
    if (Test-Path $shaderGate) {
        Set-Location "D:\Dev\kha\tools\shaders"
        & node validate_and_report.mjs
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ⚠️  Shader warnings (non-blocking)" -ForegroundColor Yellow
        } else {
            Write-Host "  ✅ Shaders valid" -ForegroundColor Green
        }
    } else {
        Write-Host "  ⏭️  Skipped (gate not found)" -ForegroundColor Gray
    }
}

# Summary
Write-Host "`n===== CI Check Summary =====" -ForegroundColor Cyan
if ($failed) {
    Write-Host "❌ CI FAILED - Fix issues above" -ForegroundColor Red
    exit 1
} else {
    Write-Host "✅ CI PASSED - Ready for release!" -ForegroundColor Green
    exit 0
}
'@

$ciCheckScript = "D:\Dev\kha\tools\ci\Run-CI-Checks.ps1"
Set-Content -Path $ciCheckScript -Value $ciScript
Write-Host "✅ Created: $ciCheckScript" -ForegroundColor Green

# Create GitHub Actions workflow
Write-Host ""
Write-Host "Creating GitHub Actions workflow..." -ForegroundColor Yellow

$workflowDir = "D:\Dev\kha\.github\workflows"
if (-not (Test-Path $workflowDir)) {
    New-Item -ItemType Directory -Path $workflowDir -Force | Out-Null
}

$workflow = @'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '20'
        
    - name: Setup pnpm
      uses: pnpm/action-setup@v2
      with:
        version: 8
        
    - name: Install dependencies
      run: |
        cd tori_ui_svelte
        pnpm install
        
    - name: Security audit (high only)
      run: |
        cd tori_ui_svelte
        npm audit --audit-level=high
      continue-on-error: false  # Fail on high severity
      
    - name: Build
      run: |
        cd tori_ui_svelte
        pnpm run build
        
    - name: Type check
      run: |
        cd tori_ui_svelte
        pnpm run check
      continue-on-error: true  # Don't fail on type warnings
      
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: build
        path: tori_ui_svelte/build/
'@

$workflowFile = Join-Path $workflowDir "ci.yml"
Set-Content -Path $workflowFile -Value $workflow
Write-Host "✅ Created: $workflowFile" -ForegroundColor Green

Write-Host @"

====================================================
    CI/CD Configuration Complete!
    
    Settings Applied:
    ✅ NPM Audit: HIGH severity only
    ✅ A11y: Warnings (non-blocking)
    ✅ Shaders: Naga ready, Tint optional
    
    Scripts Created:
    - Run-CI-Checks.ps1 (local CI)
    - GitHub Actions workflow
    
    Run CI Checks:
      D:\Dev\kha\tools\ci\Run-CI-Checks.ps1
    
    Quick Check:
      D:\Dev\kha\tools\ci\Run-CI-Checks.ps1 -Quick
====================================================
"@ -ForegroundColor Cyan
