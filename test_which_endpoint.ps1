# Quick test to see which init endpoint is being used

Write-Host "Testing which init endpoint is active..." -ForegroundColor Yellow

# Test with user_id (expected by router)
Write-Host "`n1. Testing with user_id field:" -ForegroundColor Cyan
$body1 = @{ user_id = "test1" } | ConvertTo-Json
try {
    $response1 = Invoke-WebRequest -Uri "http://localhost:8002/api/soliton/init" `
        -Method Post -ContentType "application/json" -Body $body1 -ErrorAction Stop
    Write-Host "✅ Success with user_id" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed with user_id: $($_.Exception.Message)" -ForegroundColor Red
}

# Test with user (expected by duplicate endpoint)
Write-Host "`n2. Testing with user field:" -ForegroundColor Cyan
$body2 = @{ user = "test2" } | ConvertTo-Json
try {
    $response2 = Invoke-WebRequest -Uri "http://localhost:8002/api/soliton/init" `
        -Method Post -ContentType "application/json" -Body $body2 -ErrorAction Stop
    Write-Host "✅ Success with user" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed with user: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host "`nConclusion:" -ForegroundColor Yellow
Write-Host "The endpoint that accepts the request is the one being used."
