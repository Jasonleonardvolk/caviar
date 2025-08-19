# Test Soliton Endpoints with curl
# Run this after starting TORI with: python enhanced_launcher.py

Write-Host "`n=== Testing Soliton API Endpoints ===" -ForegroundColor Blue

# Test 1: Health Check
Write-Host "`n1. Testing API Health:" -ForegroundColor Yellow
curl -s http://localhost:8002/api/health | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Test 2: Soliton Init
Write-Host "`n2. Testing Soliton Init (POST /api/soliton/init):" -ForegroundColor Yellow
$initResponse = curl -s -X POST http://localhost:8002/api/soliton/init `
    -H "Content-Type: application/json" `
    -d '{"user_id": "test_user"}'
    
Write-Host "Response:" -ForegroundColor Green
$initResponse | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Test 3: Soliton Stats
Write-Host "`n3. Testing Soliton Stats (GET /api/soliton/stats/adminuser):" -ForegroundColor Yellow
$statsResponse = curl -s http://localhost:8002/api/soliton/stats/adminuser

Write-Host "Response:" -ForegroundColor Green
$statsResponse | ConvertFrom-Json | ConvertTo-Json -Depth 10

# Test 4: Check OpenAPI docs for routes
Write-Host "`n4. Checking if Soliton routes are registered:" -ForegroundColor Yellow
$openapi = curl -s http://localhost:8002/openapi.json | ConvertFrom-Json
$solitonPaths = $openapi.paths.PSObject.Properties | Where-Object { $_.Name -like "*soliton*" }

if ($solitonPaths) {
    Write-Host "Found Soliton endpoints:" -ForegroundColor Green
    foreach ($path in $solitonPaths) {
        Write-Host "  - $($path.Name)" -ForegroundColor Cyan
    }
} else {
    Write-Host "No Soliton endpoints found in OpenAPI!" -ForegroundColor Red
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Blue
