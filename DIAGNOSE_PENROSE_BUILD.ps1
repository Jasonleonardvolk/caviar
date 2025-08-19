# Penrose Build Diagnostic Script - Fixed Version
Write-Host "Penrose Build Diagnostics" -ForegroundColor Cyan
Write-Host "========================" -ForegroundColor Cyan

# Check Python version and environment
Write-Host "`n[1] Python Environment:" -ForegroundColor Yellow
python --version
Write-Host "Python path: $((Get-Command python).Path)"
Write-Host "Pip version:"
pip --version

# Check if we're in a virtual environment
if ($env:VIRTUAL_ENV) {
    Write-Host "[OK] Virtual environment active: $env:VIRTUAL_ENV" -ForegroundColor Green
} else {
    Write-Host "[WARNING] No virtual environment detected" -ForegroundColor Yellow
}

# Check Rust installation
Write-Host "`n[2] Rust Installation:" -ForegroundColor Yellow
$rustInstalled = $false
try {
    $rustVersion = rustc --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Rust installed: $rustVersion" -ForegroundColor Green
        $rustInstalled = $true
        
        # Check toolchain
        Write-Host "Active toolchain:"
        rustup show
        
        # Check if MSVC toolchain is installed
        Write-Host "`nChecking for MSVC toolchain:"
        rustup toolchain list | Select-String "msvc"
    } else {
        Write-Host "[ERROR] Rust not found" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Rust not found: $_" -ForegroundColor Red
}

# Check Visual Studio / Build Tools
Write-Host "`n[3] Visual Studio / Build Tools:" -ForegroundColor Yellow
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    Write-Host "Visual Studio installations found:"
    try {
        & $vsWhere -products * -format json | ConvertFrom-Json | ForEach-Object {
            Write-Host "  - $($_.displayName) at $($_.installationPath)"
        }
    } catch {
        Write-Host "[ERROR] Could not parse VS installations" -ForegroundColor Red
    }
} else {
    Write-Host "[WARNING] vswhere not found - Visual Studio may not be installed" -ForegroundColor Yellow
}

# Check for cl.exe (C++ compiler)
Write-Host "`nChecking for C++ compiler (cl.exe):"
$clFound = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clFound) {
    Write-Host "[OK] cl.exe found at: $($clFound.Path)" -ForegroundColor Green
} else {
    Write-Host "[ERROR] cl.exe not found - C++ build tools not in PATH" -ForegroundColor Red
    Write-Host "You may need to:" -ForegroundColor Yellow
    Write-Host "  1. Install Visual Studio Build Tools from:" -ForegroundColor White
    Write-Host "     https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    Write-Host "  2. Select 'Desktop development with C++' workload during installation" -ForegroundColor White
}

# Check maturin
Write-Host "`n[4] Maturin:" -ForegroundColor Yellow
try {
    $maturinInstalled = pip show maturin 2>&1 | Out-String
    if ($maturinInstalled -match "Name: maturin") {
        Write-Host "[OK] Maturin installed" -ForegroundColor Green
        $version = if ($maturinInstalled -match "Version: (.+)") { $matches[1] } else { "unknown" }
        Write-Host "Version: $version" -ForegroundColor Gray
    } else {
        Write-Host "[ERROR] Maturin not installed" -ForegroundColor Red
    }
} catch {
    Write-Host "[ERROR] Could not check maturin: $_" -ForegroundColor Red
}

# Check required Python packages
Write-Host "`n[5] Python Dependencies:" -ForegroundColor Yellow
$packages = @("numpy", "setuptools", "wheel")
foreach ($package in $packages) {
    $output = pip show $package 2>&1 | Out-String
    if ($output -match "Name: $package") {
        $version = if ($output -match "Version: (.+)") { $matches[1] } else { "unknown" }
        Write-Host "[OK] $package installed (version $version)" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] $package not installed" -ForegroundColor Red
    }
}

# Try a simple maturin build to get specific error
if ($rustInstalled) {
    Write-Host "`n[6] Attempting minimal build to capture errors:" -ForegroundColor Yellow
    $penroseDir = Join-Path $PSScriptRoot "concept_mesh\penrose_rs"
    
    if (Test-Path $penroseDir) {
        Push-Location $penroseDir
        Write-Host "Building in: $penroseDir" -ForegroundColor Gray
        
        # First, try to create a simple setup
        Write-Host "`nRunning maturin build (this will show specific errors):" -ForegroundColor Yellow
        maturin build --interpreter python 2>&1 | Out-String | Write-Host
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`n[ERROR] Build failed with exit code: $LASTEXITCODE" -ForegroundColor Red
            Write-Host "Key errors to look for:" -ForegroundColor Yellow
            Write-Host "  - 'error: Microsoft Visual C++ 14.0 is required'" -ForegroundColor White
            Write-Host "  - 'error: linker link.exe not found'" -ForegroundColor White
            Write-Host "  - 'error: could not compile'" -ForegroundColor White
        }
        
        Pop-Location
    } else {
        Write-Host "[ERROR] Penrose directory not found at: $penroseDir" -ForegroundColor Red
    }
}

# Recommendations
Write-Host "`n[7] Recommendations:" -ForegroundColor Yellow

if (-not $rustInstalled) {
    Write-Host "`nTo install Rust:" -ForegroundColor White
    Write-Host "  1. Download from: https://rustup.rs/" -ForegroundColor Cyan
    Write-Host "  2. Run the installer and follow prompts" -ForegroundColor White
    Write-Host "  3. Restart this terminal after installation" -ForegroundColor White
}

if (-not $clFound) {
    Write-Host "`nTo install C++ Build Tools:" -ForegroundColor White
    Write-Host "  1. Download Build Tools from:" -ForegroundColor White
    Write-Host "     https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022" -ForegroundColor Cyan
    Write-Host "  2. Run installer and select 'Desktop development with C++'" -ForegroundColor White
    Write-Host "  3. Restart this terminal after installation" -ForegroundColor White
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")