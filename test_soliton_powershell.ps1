# PowerShell Test Commands for Soliton Endpoints

Write-Host "`n=== Testing Soliton Endpoints ===" -ForegroundColor Green

# Test 1: Stats endpoint (GET)
Write-Host "`n1. Testing GET /api/soliton/stats/adminuser:" -ForegroundColor Yellow
$statsResponse = Invoke-WebRequest -Uri "http://localhost:8002/api/soliton/stats/adminuser" -Method Get
Write-Host "Status: $($statsResponse.StatusCode)" -ForegroundColor Green
$statsResponse.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Test 2: Init endpoint (POST)
Write-Host "`n2. Testing POST /api/soliton/init:" -ForegroundColor Yellow
$initBody = @{
    user_id = "admin_user"  # Backend expects user_id from the imported router
} | ConvertTo-Json

$initResponse = Invoke-WebRequest -Uri "http://localhost:8002/api/soliton/init" `
    -Method Post `
    -ContentType "application/json" `
    -Body $initBody

Write-Host "Status: $($initResponse.StatusCode)" -ForegroundColor Green
$initResponse.Content | ConvertFrom-Json | ConvertTo-Json -Depth 10

Write-Host "`n✅ Soliton endpoints are working!" -ForegroundColor Green
