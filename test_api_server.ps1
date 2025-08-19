# Test script for TORI API server
# Run this after starting TORI to verify the API is working

param(
    [int]$Port = 8002
)

Write-Host "Testing TORI API Server on port $Port..." -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Test 1: Health Check
Write-Host "`nTest 1: Health Check" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$Port/api/health" -Method GET -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Health check passed!" -ForegroundColor Green
        $health = $response.Content | ConvertFrom-Json
        Write-Host "  Status: $($health.status)" -ForegroundColor Gray
        Write-Host "  Timestamp: $($health.timestamp)" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ Health check failed: $_" -ForegroundColor Red
}

# Test 2: Root endpoint
Write-Host "`nTest 2: Root Endpoint" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$Port/" -Method GET -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Root endpoint accessible!" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ Root endpoint failed: $_" -ForegroundColor Red
}

# Test 3: System Status
Write-Host "`nTest 3: System Status" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$Port/api/system/status" -Method GET -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ System status accessible!" -ForegroundColor Green
        $status = $response.Content | ConvertFrom-Json
        Write-Host "  System healthy: $($status.healthy)" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ System status failed: $_" -ForegroundColor Red
}

# Test 4: API Docs
Write-Host "`nTest 4: API Documentation" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:$Port/docs" -Method GET -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ API documentation available at http://localhost:$Port/docs" -ForegroundColor Green
    }
} catch {
    Write-Host "✗ API docs not accessible: $_" -ForegroundColor Red
}

# Test 5: Answer endpoint (POST)
Write-Host "`nTest 5: Answer Endpoint (Prajna compatibility)" -ForegroundColor Yellow
try {
    $body = @{
        user_query = "What is consciousness?"
        persona = @{
            name = "Test"
            psi = "analytical"
        }
    } | ConvertTo-Json

    $response = Invoke-WebRequest -Uri "http://localhost:$Port/api/answer" -Method POST -Body $body -ContentType "application/json" -UseBasicParsing
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Answer endpoint working!" -ForegroundColor Green
        $answer = $response.Content | ConvertFrom-Json
        Write-Host "  Response received with $($answer.response.Length) characters" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ Answer endpoint failed: $_" -ForegroundColor Red
}

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "API Server Test Complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
