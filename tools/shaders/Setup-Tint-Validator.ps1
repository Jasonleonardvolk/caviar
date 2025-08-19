#!/usr/bin/env pwsh
# Setup-Tint-Validator.ps1
# Add Tint WGSL validator alongside Naga for comprehensive shader validation

$ErrorActionPreference = "Stop"

Write-Host "===== Setting Up Tint WGSL Validator =====" -ForegroundColor Cyan
Write-Host ""
Write-Host "This adds Tint for additional WGSL validation coverage." -ForegroundColor Gray
Write-Host "You already pass with Naga, so this is optional." -ForegroundColor Yellow
Write-Host ""

$tintDir = "D:\Dev\kha\tools\shaders\bin"
$tintExe = Join-Path $tintDir "tint.exe"

# Check if Tint already exists
if (Test-Path $tintExe) {
    Write-Host "✅ Tint already installed at: $tintExe" -ForegroundColor Green
    exit 0
}

Write-Host "Tint provides additional WGSL validation beyond Naga:" -ForegroundColor Yellow
Write-Host "  - More detailed error messages" -ForegroundColor Gray
Write-Host "  - Stricter spec compliance" -ForegroundColor Gray
Write-Host "  - Additional optimization passes" -ForegroundColor Gray
Write-Host ""

# Download options
Write-Host "To install Tint, choose an option:" -ForegroundColor Cyan
Write-Host ""
Write-Host "Option 1: Download from Dawn project (recommended)" -ForegroundColor Yellow
Write-Host "  Visit: https://dawn.googlesource.com/dawn" -ForegroundColor Gray
Write-Host "  Or: https://github.com/google/dawn-builds/releases" -ForegroundColor Gray
Write-Host ""
Write-Host "Option 2: Build from source" -ForegroundColor Yellow
Write-Host "  git clone https://dawn.googlesource.com/dawn" -ForegroundColor Gray
Write-Host "  Follow build instructions for Windows" -ForegroundColor Gray
Write-Host ""
Write-Host "Option 3: Use automated download (if available)" -ForegroundColor Yellow

$choice = Read-Host "Download Tint automatically? (y/n)"

if ($choice -eq 'y') {
    Write-Host ""
    Write-Host "Attempting to download Tint..." -ForegroundColor Yellow
    
    # Create bin directory if it doesn't exist
    if (-not (Test-Path $tintDir)) {
        New-Item -ItemType Directory -Path $tintDir -Force | Out-Null
    }
    
    # Try to download from GitHub releases
    $downloadUrl = "https://github.com/google/dawn-builds/releases/latest/download/tint-windows-x64.exe"
    $tempFile = Join-Path $env:TEMP "tint.exe"
    
    try {
        Write-Host "Downloading from: $downloadUrl" -ForegroundColor Gray
        Invoke-WebRequest -Uri $downloadUrl -OutFile $tempFile -ErrorAction Stop
        
        # Move to final location
        Move-Item -Path $tempFile -Destination $tintExe -Force
        Write-Host "✅ Tint downloaded successfully!" -ForegroundColor Green
        
    } catch {
        Write-Host "⚠️  Automated download failed" -ForegroundColor Yellow
        Write-Host "Please download manually from:" -ForegroundColor Gray
        Write-Host "  https://github.com/google/dawn-builds/releases" -ForegroundColor White
        Write-Host ""
        Write-Host "Place tint.exe in:" -ForegroundColor Gray
        Write-Host "  $tintDir\" -ForegroundColor White
    }
}

# Update validate_and_report.mjs to use Tint
if (Test-Path $tintExe) {
    Write-Host ""
    Write-Host "Updating shader validation script..." -ForegroundColor Yellow
    
    $validateScript = "D:\Dev\kha\tools\shaders\validate_and_report.mjs"
    
    if (Test-Path $validateScript) {
        # Create backup
        Copy-Item $validateScript "$validateScript.backup" -Force
        
        # Add Tint validation (simplified example)
        $patchContent = @'

// Add Tint validation alongside Naga
async function validateWithTint(shaderPath) {
    const tintPath = path.join(__dirname, 'bin', 'tint.exe');
    
    if (!fs.existsSync(tintPath)) {
        return { success: false, message: 'Tint not found' };
    }
    
    try {
        const result = await exec(`"${tintPath}" "${shaderPath}" --validate`);
        return { success: true, message: 'Tint validation passed' };
    } catch (error) {
        return { success: false, message: error.stderr || error.message };
    }
}

// Update main validation to include Tint
// Look for the Naga validation section and add Tint after it
'@
        
        Write-Host "  ⚠️  Please manually update validate_and_report.mjs" -ForegroundColor Yellow
        Write-Host "  to include Tint validation alongside Naga" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "===== Setup Complete =====" -ForegroundColor Cyan

if (Test-Path $tintExe) {
    Write-Host "✅ Tint is installed and ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Test it with:" -ForegroundColor Yellow
    Write-Host "  $tintExe <shader.wgsl> --validate" -ForegroundColor Gray
} else {
    Write-Host "ℹ️  Tint is optional - you already pass with Naga" -ForegroundColor Blue
    Write-Host "Install it later if you want additional validation coverage" -ForegroundColor Gray
}
