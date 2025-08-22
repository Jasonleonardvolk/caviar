# Start-With-Mocks.ps1
# Simple script that guarantees mocks work by setting environment correctly

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "     Starting Server with Mocks        " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill any existing process
Write-Host "[1] Stopping existing servers..." -ForegroundColor Yellow
$existingProcess = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($existingProcess) {
    $procId = $existingProcess.OwningProcess
    if ($procId -gt 0) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Write-Host "  Stopped process on port 3000" -ForegroundColor Green
        Start-Sleep -Seconds 1
    }
}

# Step 2: Ensure .env.local has correct values
Write-Host "`n[2] Ensuring .env.local is correct..." -ForegroundColor Yellow
$envContent = @"
PORT=3000
IRIS_USE_MOCKS=1
IRIS_ALLOW_UNAUTH=1
NODE_ENV=production
LOCAL_UPLOAD_DIR=var/uploads
"@

$envContent | Out-File -FilePath ".env.local" -Encoding UTF8
Write-Host "  Created .env.local with IRIS_USE_MOCKS=1" -ForegroundColor Green

# Step 3: Check if build exists
Write-Host "`n[3] Checking build..." -ForegroundColor Yellow
if (-not (Test-Path "build\index.js")) {
    Write-Host "  Build doesn't exist! Building now..." -ForegroundColor Yellow
    
    # Clear cache first
    if (Test-Path ".\.svelte-kit") {
        Remove-Item ".\.svelte-kit" -Recurse -Force
        Write-Host "  Cleared .svelte-kit cache" -ForegroundColor Gray
    }
    
    # Build with mocks enabled
    $env:IRIS_USE_MOCKS = "1"
    & pnpm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Build complete!" -ForegroundColor Green
} else {
    Write-Host "  Build exists" -ForegroundColor Green
}

# Step 4: Start server with explicit environment
Write-Host "`n[4] Starting server with mocks..." -ForegroundColor Yellow

# Method 1: Direct node with env vars inline (most reliable)
Write-Host "  Using direct node execution with inline env vars..." -ForegroundColor Gray

$startScript = @"
`$env:IRIS_USE_MOCKS = '1'
`$env:PORT = '3000'
`$env:IRIS_ALLOW_UNAUTH = '1'
`$env:NODE_ENV = 'production'
Set-Location '$PWD'
Write-Host 'Starting with IRIS_USE_MOCKS=' -NoNewline
Write-Host `$env:IRIS_USE_MOCKS -ForegroundColor Green
node build/index.js
"@

$job = Start-Job -ScriptBlock ([scriptblock]::Create($startScript)) -Name "iris-mock"

# Wait for server to start
Write-Host "  Waiting for server to start..."
$maxWait = 20
$waited = 0
while ($waited -lt $maxWait) {
    $listening = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        Write-Host "  Server is listening on port 3000!" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 1
    $waited++
    Write-Host "." -NoNewline
}
Write-Host ""

if ($waited -ge $maxWait) {
    Write-Host "  Server failed to start!" -ForegroundColor Red
    Stop-Job -Name "iris-mock" -ErrorAction SilentlyContinue
    Remove-Job -Name "iris-mock" -ErrorAction SilentlyContinue
    exit 1
}

# Step 5: Test the endpoints
Write-Host "`n[5] Testing mock endpoints..." -ForegroundColor Yellow

Start-Sleep -Seconds 2  # Give server time to initialize

$tests = @(
    @{ Name = "Health"; Url = "http://127.0.0.1:3000/api/health" },
    @{ Name = "PDF Stats"; Url = "http://127.0.0.1:3000/api/pdf/stats" },
    @{ Name = "Memory State"; Url = "http://127.0.0.1:3000/api/memory/state" }
)

$allPass = $true
foreach ($test in $tests) {
    try {
        $response = Invoke-RestMethod -Uri $test.Url -Method GET
        
        if ($test.Name -eq "Health") {
            if ($response.status -eq "ok" -or $response.ok) {
                Write-Host "  [PASS] $($test.Name)" -ForegroundColor Green
            } else {
                Write-Host "  [FAIL] $($test.Name)" -ForegroundColor Red
                $allPass = $false
            }
        } else {
            if ($response.note -eq 'mock') {
                Write-Host "  [PASS] $($test.Name) - Mock confirmed!" -ForegroundColor Green
                Write-Host "        $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
            } elseif ($response.error -like "*not configured*") {
                Write-Host "  [FAIL] $($test.Name) - Still 'not configured'" -ForegroundColor Red
                Write-Host "        $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
                $allPass = $false
            } else {
                Write-Host "  [WARN] $($test.Name) - Unexpected response" -ForegroundColor Yellow
                Write-Host "        $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
            }
        }
    } catch {
        Write-Host "  [ERROR] $($test.Name) - $_" -ForegroundColor Red
        $allPass = $false
    }
}

# Summary
Write-Host "`n========================================" -ForegroundColor Cyan
if ($allPass) {
    Write-Host "       SUCCESS! Mocks Working!         " -ForegroundColor Green
} else {
    Write-Host "      PROBLEM: Mocks Not Working      " -ForegroundColor Red
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Server running at: http://127.0.0.1:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Commands:" -ForegroundColor Yellow
Write-Host "  Get-Job iris-mock         # Check status" -ForegroundColor Gray
Write-Host "  Receive-Job iris-mock     # View logs" -ForegroundColor Gray
Write-Host "  Stop-Job iris-mock        # Stop server" -ForegroundColor Gray
Write-Host "  Remove-Job iris-mock      # Clean up" -ForegroundColor Gray

if (-not $allPass) {
    Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check server logs: Receive-Job iris-mock" -ForegroundColor Gray
    Write-Host "  2. Run diagnostics: .\Debug-Mock-Issue.ps1" -ForegroundColor Gray
    Write-Host "  3. Force rebuild: .\Force-Fresh-Build.ps1" -ForegroundColor Gray
}

Write-Host "`nDone." -ForegroundColor Green
