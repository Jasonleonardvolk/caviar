#!/usr/bin/env pwsh
# Install TORI in editable mode for proper imports

Write-Host "`n=== Installing TORI Package ===" -ForegroundColor Cyan
Write-Host "This will make all imports work correctly" -ForegroundColor Gray

# Navigate to project root
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Install in editable mode
Write-Host "`n[*] Installing package in editable mode..." -ForegroundColor Yellow
pip install -e .

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[OK] Package installed successfully!" -ForegroundColor Green
    
    # Test imports
    Write-Host "`n[*] Testing imports..." -ForegroundColor Yellow
    python -c "import mcp_metacognitive.core.soliton_memory as sm; print('[OK] soliton_memory imported from', sm.__file__)"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ All imports working correctly!" -ForegroundColor Green
        Write-Host "`nYou can now run:" -ForegroundColor Cyan
        Write-Host "  python -m mcp_metacognitive.server" -ForegroundColor White
        Write-Host "  OR" -ForegroundColor Gray
        Write-Host "  uvicorn mcp_metacognitive.server:app --host 0.0.0.0 --port 8100" -ForegroundColor White
    } else {
        Write-Host "`n❌ Import test failed" -ForegroundColor Red
    }
} else {
    Write-Host "`n❌ Installation failed" -ForegroundColor Red
}

# Show Python path
Write-Host "`n[*] Current Python path:" -ForegroundColor Yellow
python -c "import sys, pprint; pprint.pp(sys.path[:5])"
