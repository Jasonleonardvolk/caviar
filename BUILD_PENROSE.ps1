# Build Penrose Engine Rust Extension
# PowerShell script for building and installing penrose_engine_rs

param(
    [switch]$Development = $false,
    [switch]$Clean = $false,
    [switch]$Test = $false
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-ColorHost($Text, $Color) {
    Write-Host $Text -ForegroundColor $Color
}

Write-ColorHost "========================================" "Cyan"
Write-ColorHost "  Penrose Engine Rust Build Script" "Cyan"
Write-ColorHost "========================================" "Cyan"

# Navigate to project directory
$projectPath = "C:\Users\jason\Desktop\tori\kha\concept_mesh\penrose_rs"
Write-ColorHost "`nNavigating to: $projectPath" "Yellow"
Set-Location $projectPath

# Clean if requested
if ($Clean) {
    Write-ColorHost "`nCleaning previous builds..." "Yellow"
    Remove-Item -Recurse -Force target -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force *.egg-info -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
    Write-ColorHost "Clean complete!" "Green"
}

# Check prerequisites
Write-ColorHost "`nChecking prerequisites..." "Yellow"

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-ColorHost "Found: $pythonVersion" "Green"
} catch {
    Write-ColorHost "ERROR: Python not found. Please install Python 3.9+" "Red"
    exit 1
}

# Check Rust
try {
    $rustVersion = rustc --version 2>&1
    Write-ColorHost "Found: $rustVersion" "Green"
} catch {
    Write-ColorHost "ERROR: Rust not found. Please install from https://rustup.rs/" "Red"
    exit 1
}

# Check Maturin
try {
    $maturinVersion = maturin --version 2>&1
    Write-ColorHost "Found: Maturin $maturinVersion" "Green"
} catch {
    Write-ColorHost "Maturin not found. Installing..." "Yellow"
    pip install maturin>=1.4,<2.0
}

# Build the extension
Write-ColorHost "`nBuilding Penrose Engine..." "Yellow"

if ($Development) {
    Write-ColorHost "Building in development mode (editable install)..." "Cyan"
    maturin develop
} else {
    Write-ColorHost "Building release version..." "Cyan"
    maturin build --release
    
    # Install the wheel
    Write-ColorHost "`nInstalling built wheel..." "Yellow"
    $wheel = Get-ChildItem target\wheels\*.whl | Select-Object -First 1
    if ($wheel) {
        pip install --force-reinstall $wheel.FullName
        Write-ColorHost "Successfully installed: $($wheel.Name)" "Green"
    } else {
        Write-ColorHost "ERROR: Build failed - no wheel found" "Red"
        exit 1
    }
}

# Run tests if requested
if ($Test) {
    Write-ColorHost "`nRunning tests..." "Yellow"
    
    $testScript = @'
import penrose_engine_rs
import numpy as np

print("Testing Penrose Engine...")

# Initialize engine
result = penrose_engine_rs.initialize_engine(
    max_threads=4,
    cache_size_mb=512,
    enable_gpu=False,
    precision="float32"
)
print(f"Engine initialized: {result}")

# Get engine info
info = penrose_engine_rs.get_engine_info()
print(f"Engine info: {info}")

# Test similarity computation
v1 = [1.0, 2.0, 3.0]
v2 = [4.0, 5.0, 6.0]
similarity = penrose_engine_rs.compute_similarity(v1, v2)
print(f"Similarity test: {similarity:.4f}")

# Test batch similarity
query = [1.0, 0.0, 0.0]
corpus = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
similarities = penrose_engine_rs.batch_similarity(query, corpus)
print(f"Batch similarities: {similarities}")

# Test lattice evolution
lattice = [[1.0, 2.0], [3.0, 4.0]]
phase_field = [[0.0, 0.5], [1.0, 1.5]]
curvature_field = [[0.1, 0.2], [0.3, 0.4]]
evolved = penrose_engine_rs.evolve_lattice_field(lattice, phase_field, curvature_field, 0.01)
print(f"Evolved lattice: {evolved}")

print("\nAll tests passed!")
'@
    
    $testScript | python
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorHost "`nAll tests passed!" "Green"
    } else {
        Write-ColorHost "`nTests failed!" "Red"
        exit 1
    }
}

Write-ColorHost "`n========================================" "Cyan"
Write-ColorHost "  Build Complete!" "Green"
Write-ColorHost "========================================" "Cyan"

# Display usage
Write-Host "`nUsage in Python:"
Write-Host "  import penrose_engine_rs"
Write-Host "  penrose_engine_rs.initialize_engine(4, 512, False, 'float32')"
Write-Host ""
