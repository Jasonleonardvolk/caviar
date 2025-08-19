# Bulletproof Penrose Build Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Bulletproof Penrose Build (PowerShell)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$projectRoot = $PWD.Path

# Check if in correct directory
if (-not (Test-Path "concept_mesh\penrose_rs\Cargo.toml")) {
    Write-Host "ERROR: Run from project root (C:\Users\jason\Desktop\tori\kha)" -ForegroundColor Red
    exit 1
}

# Step 1: Create venv if needed
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Step 2: Get venv Python path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
Write-Host "Virtual environment Python: $venvPython" -ForegroundColor Green

# Step 3: Install maturin using venv Python
Write-Host "`nInstalling/updating maturin..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade maturin

# Step 4: Build with explicit interpreter
Write-Host "`nBuilding Penrose engine..." -ForegroundColor Yellow
Push-Location "concept_mesh\penrose_rs"
& maturin develop --release -i $venvPython
Pop-Location

# Step 5: Test import using venv Python
Write-Host "`nTesting import..." -ForegroundColor Yellow
$testResult = & $venvPython -c "import penrose_engine_rs as p; print('SUCCESS: Penrose Rust backend ready -', p.__name__)" 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host $testResult -ForegroundColor Green
    Write-Host "`nThe ModuleNotFoundError is fixed!" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. git add/commit/push" -ForegroundColor White
    Write-Host "2. Watch CI turn green" -ForegroundColor White
} else {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "BUILD FAILED!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host $testResult -ForegroundColor Red
}

Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
