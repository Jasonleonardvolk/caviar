#!/usr/bin/env pwsh
# Test Soliton Init Fix - Final Version
# =====================================

Write-Host "`n=== Testing Soliton Init Fix (Final) ===" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Gray

# Test 1: Test Soliton init with CORRECT field name
Write-Host "`n[1] Testing Soliton init endpoint with correct field name..." -ForegroundColor Yellow
$initPayload = @{
    userId = "adminuser"  # Changed from user_id to userId
} | ConvertTo-Json

Write-Host "Sending payload: $initPayload" -ForegroundColor Gray

try {
    $initResponse = Invoke-RestMethod -Method Post -Uri 'http://localhost:8002/api/soliton/init' `
        -ContentType 'application/json' `
        -Body $initPayload
    
    Write-Host "[OK] Soliton init successful!" -ForegroundColor Green
    Write-Host "  Success: $($initResponse.success)" -ForegroundColor White
    Write-Host "  Engine: $($initResponse.engine)" -ForegroundColor White
    Write-Host "  User ID: $($initResponse.user_id)" -ForegroundColor White
    Write-Host "  Message: $($initResponse.message)" -ForegroundColor White
    Write-Host "  Lattice Ready: $($initResponse.lattice_ready)" -ForegroundColor White
} catch {
    Write-Host "[ERROR] Soliton init failed!" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "The issue was a field name mismatch:" -ForegroundColor Yellow
Write-Host "- Model expects: userId (camelCase)" -ForegroundColor White
Write-Host "- Code was trying: user_id (snake_case)" -ForegroundColor White
Write-Host "- Fixed both the code and the request payload" -ForegroundColor White

Write-Host "`nRestart the server to apply the fix!" -ForegroundColor Cyan
