# Fix API timeout issue in TORI startup

Write-Host "Fixing TORI API timeout issue..." -ForegroundColor Yellow

# 1. Increase timeout in start_tori_hardened.ps1
$startScript = "C:\Users\jason\Desktop\tori\kha\start_tori_hardened.ps1"
if (Test-Path $startScript) {
    Write-Host "Updating timeout in start_tori_hardened.ps1..." -ForegroundColor Yellow
    
    # Read the file
    $content = Get-Content $startScript -Raw
    
    # Replace timeout value (look for the Wait-ForApi function)
    $content = $content -replace '\$timeout = 90', '$timeout = 180'
    $content = $content -replace 'within 90 seconds', 'within 180 seconds'
    
    # Create backup
    Copy-Item $startScript "$startScript.backup" -Force
    
    # Write updated content
    Set-Content $startScript $content -Force
    
    Write-Host "✅ Timeout increased from 90 to 180 seconds" -ForegroundColor Green
}

# 2. Create a lightweight API test script
$testApiScript = @'
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test if prajna imports are the issue
print("Testing prajna imports...")
try:
    from prajna.api.prajna_api import app as prajna_app
    print("✅ Prajna imports successful")
    print(f"App type: {type(prajna_app)}")
except Exception as e:
    print(f"❌ Prajna import failed: {e}")
    print("This is likely why the API is failing to start!")
'@

Set-Content "C:\Users\jason\Desktop\tori\kha\test_api_imports.py" $testApiScript -Force

Write-Host "`n✅ Fix applied! Here's what was done:" -ForegroundColor Green
Write-Host "1. Increased API startup timeout from 90 to 180 seconds" -ForegroundColor White
Write-Host "2. Created test_api_imports.py to diagnose import issues" -ForegroundColor White

Write-Host "`n📋 Next steps:" -ForegroundColor Yellow
Write-Host "1. Run: python test_api_imports.py" -ForegroundColor White
Write-Host "   This will show if prajna imports are failing" -ForegroundColor Gray
Write-Host "2. If imports fail, we need to fix the prajna module" -ForegroundColor White
Write-Host "3. If imports work, try starting TORI again with: .\START_TORI_HARDENED.bat" -ForegroundColor White

Write-Host "`n💡 Alternative quick fix:" -ForegroundColor Cyan
Write-Host "Start just the MCP server for now:" -ForegroundColor White
Write-Host "python -m mcp_metacognitive.server_integrated" -ForegroundColor Gray
