# Penrose Engine Dependencies - PowerShell Script

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Installing Penrose Dependencies" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to install a package
function Install-Package {
    param(
        [string]$PackageName,
        [string]$Description,
        [bool]$Optional = $false
    )
    
    Write-Host "[Installing] $PackageName ($Description)..." -ForegroundColor Yellow
    
    & python -m pip install $PackageName
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: $PackageName installed" -ForegroundColor Green
        return $true
    } else {
        if ($Optional) {
            Write-Host "WARNING: Failed to install $PackageName (optional)" -ForegroundColor Yellow
            Write-Host "         Penrose will work without it, but with reduced performance" -ForegroundColor Yellow
            return $false
        } else {
            Write-Host "ERROR: Failed to install $PackageName (required)" -ForegroundColor Red
            return $false
        }
    }
}

# Install dependencies
$success = $true

Write-Host "`n[1/3] scipy - Required for sparse matrix operations" -ForegroundColor White
if (-not (Install-Package -PackageName "scipy" -Description "sparse matrix operations")) {
    $success = $false
}

Write-Host "`n[2/3] zstandard - Required for compression" -ForegroundColor White
if (-not (Install-Package -PackageName "zstandard" -Description "compression")) {
    $success = $false
}

Write-Host "`n[3/3] numba - Optional JIT compilation (2x speedup)" -ForegroundColor White
$numbaInstalled = Install-Package -PackageName "numba" -Description "JIT compilation" -Optional $true

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   Installation Summary" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if ($success) {
    Write-Host "✅ All required dependencies installed!" -ForegroundColor Green
    
    if ($numbaInstalled) {
        Write-Host "✅ Numba JIT compilation enabled (maximum performance)" -ForegroundColor Green
    } else {
        Write-Host "⚠️  Numba not installed (Penrose will work but ~2x slower)" -ForegroundColor Yellow
    }
    
    Write-Host "`nExpected Performance:" -ForegroundColor Cyan
    Write-Host "  • 22,000x speedup for similarity computations" -ForegroundColor White
    Write-Host "  • Memory usage: ~0.3% of dense matrix" -ForegroundColor White
    Write-Host "  • Compression: 4-5x reduction in storage" -ForegroundColor White
    
    Write-Host "`n🧪 Verifying installation..." -ForegroundColor Yellow
    & python verify_penrose.py
    
} else {
    Write-Host "❌ Failed to install required dependencies" -ForegroundColor Red
    Write-Host "`nPlease try manually:" -ForegroundColor Yellow
    Write-Host "  pip install scipy zstandard numba" -ForegroundColor White
}

Write-Host "`nPress any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
