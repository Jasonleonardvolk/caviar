Write-Host "`n🔧 Fixing Poetry scipy Installation Issue" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Kill all Python processes
Write-Host "`n1️⃣ Stopping all Python processes..." -ForegroundColor Yellow
Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process ipython* -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process jupyter* -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "✅ Python processes stopped" -ForegroundColor Green

# Step 2: Clear pip cache
Write-Host "`n2️⃣ Clearing pip cache..." -ForegroundColor Yellow
pip cache purge
Write-Host "✅ Pip cache cleared" -ForegroundColor Green

# Step 3: Try to remove scipy manually
Write-Host "`n3️⃣ Attempting to remove scipy manually..." -ForegroundColor Yellow
$scipyPath = "C:\Users\jason\Desktop\tori\kha\.venv\Lib\site-packages\scipy"
$scipyInfoPath = "C:\Users\jason\Desktop\tori\kha\.venv\Lib\site-packages\scipy-*.dist-info"

if (Test-Path $scipyPath) {
    try {
        # Take ownership first (requires admin)
        takeown /f $scipyPath /r /d y 2>$null
        icacls $scipyPath /grant "${env:USERNAME}:F" /t /q 2>$null
        
        # Remove read-only attributes
        Get-ChildItem -Path $scipyPath -Recurse | ForEach-Object {
            $_.Attributes = 'Normal'
        }
        
        # Delete the directory
        Remove-Item -Path $scipyPath -Recurse -Force -ErrorAction Stop
        Remove-Item -Path $scipyInfoPath -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "✅ scipy removed successfully" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Could not remove scipy automatically: $_" -ForegroundColor Yellow
        Write-Host "Try running PowerShell as Administrator" -ForegroundColor Yellow
    }
} else {
    Write-Host "scipy directory not found" -ForegroundColor Gray
}

# Step 4: Clean __pycache__ directories
Write-Host "`n4️⃣ Cleaning __pycache__ directories..." -ForegroundColor Yellow
Get-ChildItem -Path ".venv" -Include "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "✅ Cache directories cleaned" -ForegroundColor Green

Write-Host "`n✨ Cleanup complete! Now try:" -ForegroundColor Green
Write-Host "  poetry install" -ForegroundColor White
Write-Host "`nIf it still fails, try:" -ForegroundColor Yellow
Write-Host "  1. Close all IDEs (VS Code, PyCharm, etc.)" -ForegroundColor White
Write-Host "  2. Run PowerShell as Administrator" -ForegroundColor White
Write-Host "  3. Run this script again" -ForegroundColor White
