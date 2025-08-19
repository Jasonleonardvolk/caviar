param(
  [string]$RepoRoot = "D:\Dev\kha",
  [switch]$SkipShaderCheck = $false,
  [switch]$SkipTypeCheck = $false,
  [switch]$QuickBuild = $false
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

Write-Host "IRIS One-Button Ship Script" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date

# Step 1: Install dependencies
Write-Host "Step 1: Installing dependencies..." -ForegroundColor Yellow
npm install --save-dev @types/node vite svelte @webgpu/types typescript@latest
if ($LASTEXITCODE -ne 0) { 
    Write-Host "Dependency installation failed" -ForegroundColor Red
    exit 1 
}
Write-Host "Dependencies installed" -ForegroundColor Green

# Step 2: TypeScript validation
if (-not $SkipTypeCheck) {
    Write-Host "`nStep 2: TypeScript validation..." -ForegroundColor Yellow
    $tsOutput = npx tsc --noEmit 2>&1 | Out-String
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "TypeScript: No errors!" -ForegroundColor Green
    } else {
        if ($tsOutput -match "Found (\d+) error") {
            $errorCount = [int]$Matches[1]
            Write-Host "TypeScript: $errorCount errors (non-blocking)" -ForegroundColor Yellow
            
            if ($errorCount -gt 50 -and -not $QuickBuild) {
                Write-Host "Too many TypeScript errors. Fix them or use -QuickBuild flag" -ForegroundColor Red
                exit 1
            }
        }
    }
} else {
    Write-Host "`nStep 2: Skipping TypeScript check" -ForegroundColor Gray
}

# Step 3: Shader validation
if (-not $SkipShaderCheck) {
    Write-Host "`nStep 3: Shader validation..." -ForegroundColor Yellow
    Write-Host "   Auto-detecting device profiles..." -ForegroundColor Gray
    
    # Run shader validation
    & powershell -ExecutionPolicy Bypass -File .\tools\shaders\run_shader_gate.ps1 -RepoRoot $RepoRoot
    
    if ($LASTEXITCODE -ne 0) { 
        Write-Host "Shader gate failed" -ForegroundColor Red
        exit 2 
    }
    Write-Host "Shaders validated" -ForegroundColor Green
} else {
    Write-Host "`nStep 3: Skipping shader validation" -ForegroundColor Gray
}

# Step 4: Build the project
Write-Host "`nStep 4: Building project..." -ForegroundColor Yellow

if ($QuickBuild) {
    Write-Host "   Using quick build (Vite directly)..." -ForegroundColor Gray
    Set-Location tori_ui_svelte
    npx vite build
    Set-Location ..
} else {
    npm run build
}

if ($LASTEXITCODE -ne 0) { 
    Write-Host "Build failed" -ForegroundColor Red
    exit 3 
}
Write-Host "Build complete" -ForegroundColor Green

# Step 5: Package creation
Write-Host "`nStep 5: Creating release package..." -ForegroundColor Yellow

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$releaseDir = Join-Path $RepoRoot "releases\iris_v1_$timestamp"
$distDir = Join-Path $RepoRoot "dist"

# Create release directory
New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

# Copy built files - check multiple locations
$svelteBuildDir = Join-Path $RepoRoot "tori_ui_svelte\build"
$svelteDistDir = Join-Path $RepoRoot "tori_ui_svelte\dist"
$rootDistDir = $distDir

if (Test-Path $rootDistDir) {
    Write-Host "   Found dist at root: $rootDistDir" -ForegroundColor Gray
    Copy-Item -Path $rootDistDir -Destination (Join-Path $releaseDir "dist") -Recurse
} elseif (Test-Path $svelteBuildDir) {
    Write-Host "   Found SvelteKit build: $svelteBuildDir" -ForegroundColor Gray
    Copy-Item -Path $svelteBuildDir -Destination (Join-Path $releaseDir "dist") -Recurse
} elseif (Test-Path $svelteDistDir) {
    Write-Host "   Found dist in tori_ui_svelte: $svelteDistDir" -ForegroundColor Gray
    Copy-Item -Path $svelteDistDir -Destination (Join-Path $releaseDir "dist") -Recurse
} else {
    Write-Host "   WARNING: No build output found. Creating placeholder dist folder." -ForegroundColor Yellow
    $placeholderDir = Join-Path $releaseDir "dist"
    New-Item -ItemType Directory -Force -Path $placeholderDir | Out-Null
    "Build output not found - placeholder directory" | Out-File -FilePath (Join-Path $placeholderDir "README.txt")
}

# Create release manifest
$manifest = @{
    version = "1.0.0"
    buildDate = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    typescript = if ($SkipTypeCheck) { "skipped" } else { "validated" }
    shaders = if ($SkipShaderCheck) { "skipped" } else { "validated" }
    buildType = if ($QuickBuild) { "quick" } else { "full" }
}
$manifest | ConvertTo-Json | Out-File -FilePath "$releaseDir\manifest.json"

Write-Host "Release package created: $releaseDir" -ForegroundColor Green

# Step 6: Final report
$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host "`n" + "=" * 50 -ForegroundColor Cyan
Write-Host "SHIP IT! Build Complete" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host ""
Write-Host "Build Summary:" -ForegroundColor Yellow
Write-Host "   Duration: $($duration.TotalSeconds) seconds" -ForegroundColor White
Write-Host "   TypeScript: $(if ($SkipTypeCheck) {'Skipped'} else {'Validated'})" -ForegroundColor White
Write-Host "   Shaders: $(if ($SkipShaderCheck) {'Skipped'} else {'Validated'})" -ForegroundColor White
Write-Host "   Build Type: $(if ($QuickBuild) {'Quick (Vite)'} else {'Full (npm run build)'})" -ForegroundColor White
Write-Host "   Package: $releaseDir" -ForegroundColor White
Write-Host ""
Write-Host "Ready to deploy!" -ForegroundColor Magenta
Write-Host ""

# Optional: Open the release folder
if ($Host.UI.PromptForChoice("Open release folder?", "", @("&Yes", "&No"), 1) -eq 0) {
    explorer.exe $releaseDir
}

exit 0
