# Quick-Test-Mocks.ps1
# Quick script to build and test that mock endpoints work properly

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "====================================" -ForegroundColor Cyan
Write-Host "    Testing Mock Endpoints Fix     " -ForegroundColor Cyan  
Write-Host "====================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Build the application
Write-Host "[1/4] Building application..." -ForegroundColor Yellow
& pnpm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "Build successful!" -ForegroundColor Green

# Step 2: Set mock environment
Write-Host "`n[2/4] Setting mock environment..." -ForegroundColor Yellow
$env:IRIS_USE_MOCKS = "1"
$env:PORT = "3000"
$env:IRIS_ALLOW_UNAUTH = "1"
Write-Host "IRIS_USE_MOCKS = 1" -ForegroundColor Green
Write-Host "PORT = 3000" -ForegroundColor Green

# Step 3: Start server in background
Write-Host "`n[3/4] Starting server..." -ForegroundColor Yellow

# Kill any existing process on port 3000
$existingProcess = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($existingProcess) {
    $procId = $existingProcess.OwningProcess
    if ($procId -gt 0) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

# Start the server
$job = Start-Job -Name "iris-test-server" -ScriptBlock {
    Set-Location "D:\Dev\kha\tori_ui_svelte"
    $env:IRIS_USE_MOCKS = "1"
    $env:PORT = "3000"
    $env:IRIS_ALLOW_UNAUTH = "1"
    node build/index.js
}

# Wait for server to start
Write-Host "Waiting for server to start..."
$maxWait = 20
$waited = 0
while ($waited -lt $maxWait) {
    $listening = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        Write-Host "Server is listening on port 3000!" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 1
    $waited++
    Write-Host "." -NoNewline
}
Write-Host ""

if ($waited -ge $maxWait) {
    Write-Host "ERROR: Server failed to start within $maxWait seconds!" -ForegroundColor Red
    Stop-Job -Name "iris-test-server" -ErrorAction SilentlyContinue
    Remove-Job -Name "iris-test-server" -ErrorAction SilentlyContinue
    exit 1
}

# Step 4: Test endpoints
Write-Host "`n[4/4] Testing endpoints..." -ForegroundColor Yellow

$tests = @(
    @{ Name = "Health";        Url = "http://127.0.0.1:3000/api/health" },
    @{ Name = "Root";          Url = "http://127.0.0.1:3000/" },
    @{ Name = "Renderer";      Url = "http://127.0.0.1:3000/renderer" },
    @{ Name = "PDF Stats";     Url = "http://127.0.0.1:3000/api/pdf/stats" },
    @{ Name = "Memory State";  Url = "http://127.0.0.1:3000/api/memory/state" }
)

$allPassed = $true

foreach ($test in $tests) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $test.Url -Method GET -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Host "[PASS] $($test.Name): $($response.StatusCode)" -ForegroundColor Green
            
            # For API endpoints, also check the JSON response
            if ($test.Url -like "*api*") {
                $json = $response.Content | ConvertFrom-Json
                if ($json.note -eq 'mock') {
                    Write-Host "       Mock data confirmed!" -ForegroundColor Cyan
                } elseif ($json.ok -eq $true) {
                    Write-Host "       Response OK!" -ForegroundColor Cyan
                }
            }
        } else {
            Write-Host "[FAIL] $($test.Name): $($response.StatusCode)" -ForegroundColor Red
            $allPassed = $false
        }
    } catch {
        Write-Host "[FAIL] $($test.Name): ERROR - $($_.Exception.Message)" -ForegroundColor Red
        $allPassed = $false
    }
}

# Cleanup
Write-Host "`nStopping test server..." -ForegroundColor Yellow
Stop-Job -Name "iris-test-server" -ErrorAction SilentlyContinue
Remove-Job -Name "iris-test-server" -Force -ErrorAction SilentlyContinue

# Summary
Write-Host "`n====================================" -ForegroundColor Cyan
if ($allPassed) {
    Write-Host "   ALL TESTS PASSED! Ready to ship!" -ForegroundColor Green
    Write-Host "`n   Next: Run the full deployment:" -ForegroundColor Yellow
    Write-Host "   .\tools\release\Reset-And-Ship.ps1 -UsePM2" -ForegroundColor White
} else {
    Write-Host "   SOME TESTS FAILED! Check the errors above." -ForegroundColor Red
}
Write-Host "====================================" -ForegroundColor Cyan
