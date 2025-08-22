# Fix-And-Test-Mocks.ps1
# Applies the environment variable fix and tests it

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "      Fixing Mock Environment Detection        " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop any running servers
Write-Host "[1] Stopping existing servers..." -ForegroundColor Yellow
Stop-Job -Name "iris-*" -ErrorAction SilentlyContinue | Out-Null
& npx pm2 stop all --silent 2>$null

$existingProcess = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($existingProcess) {
    $procId = $existingProcess.OwningProcess
    if ($procId -gt 0) {
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Write-Host "  Stopped process on port 3000" -ForegroundColor Green
        Start-Sleep -Seconds 1
    }
}

# Step 2: Clear build cache
Write-Host "`n[2] Clearing build cache..." -ForegroundColor Yellow
if (Test-Path ".\.svelte-kit") {
    Remove-Item ".\.svelte-kit" -Recurse -Force
    Write-Host "  Cleared .svelte-kit" -ForegroundColor Green
}
if (Test-Path ".\node_modules\.vite") {
    Remove-Item ".\node_modules\.vite" -Recurse -Force
    Write-Host "  Cleared vite cache" -ForegroundColor Green
}

# Step 3: Set environment variables
Write-Host "`n[3] Setting environment variables..." -ForegroundColor Yellow
$env:IRIS_USE_MOCKS = "1"
$env:PORT = "3000"
$env:NODE_ENV = "production"
$env:IRIS_ALLOW_UNAUTH = "1"

Write-Host "  IRIS_USE_MOCKS = $env:IRIS_USE_MOCKS" -ForegroundColor Green
Write-Host "  PORT = $env:PORT" -ForegroundColor Green

# Step 4: Rebuild
Write-Host "`n[4] Building application..." -ForegroundColor Yellow
& pnpm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Build complete!" -ForegroundColor Green

# Step 5: Check compiled output
Write-Host "`n[5] Checking compiled output..." -ForegroundColor Yellow
$compiledFiles = @(
    ".\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js",
    ".\.svelte-kit\output\server\entries\endpoints\api\memory\state\_server.ts.js"
)

foreach ($file in $compiledFiles) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        if ($content -match "process\.env\.IRIS_USE_MOCKS|env\.IRIS_USE_MOCKS") {
            Write-Host "  Found environment checks in: $(Split-Path $file -Leaf)" -ForegroundColor Green
        }
    }
}

# Step 6: Start server with environment variables
Write-Host "`n[6] Starting server..." -ForegroundColor Yellow

# Create a script block that sets environment inline
$serverScript = @"
Set-Location 'D:\Dev\kha\tori_ui_svelte'
`$env:IRIS_USE_MOCKS = '1'
`$env:PORT = '3000'
`$env:IRIS_ALLOW_UNAUTH = '1'
`$env:NODE_ENV = 'production'
Write-Host 'Environment set:' -ForegroundColor Cyan
Write-Host "  IRIS_USE_MOCKS = `$env:IRIS_USE_MOCKS" -ForegroundColor Gray
Write-Host "  PORT = `$env:PORT" -ForegroundColor Gray
node build/index.js
"@

$job = Start-Job -Name "iris-fixed" -ScriptBlock ([scriptblock]::Create($serverScript))

# Wait for server
Write-Host "  Waiting for server..."
$waited = 0
while ($waited -lt 20) {
    $tcp = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
    if ($tcp) {
        Write-Host "  Server listening!" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 1
    $waited++
    Write-Host "." -NoNewline
}
Write-Host ""

if ($waited -ge 20) {
    Write-Host "ERROR: Server failed to start!" -ForegroundColor Red
    Stop-Job -Name "iris-fixed"
    Remove-Job -Name "iris-fixed"
    exit 1
}

# Step 7: Test the endpoints
Write-Host "`n[7] Testing mock endpoints..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

$tests = @(
    @{ Name = "PDF Stats"; Url = "http://127.0.0.1:3000/api/pdf/stats" },
    @{ Name = "Memory State"; Url = "http://127.0.0.1:3000/api/memory/state" },
    @{ Name = "Health"; Url = "http://127.0.0.1:3000/api/health" }
)

$passed = 0
$failed = 0

foreach ($test in $tests) {
    try {
        $response = Invoke-RestMethod -Uri $test.Url -Method GET
        
        if ($test.Name -eq "Health") {
            if ($response.status -eq "ok" -or $response.ok) {
                Write-Host "  [PASS] $($test.Name)" -ForegroundColor Green
                $passed++
            } else {
                Write-Host "  [FAIL] $($test.Name)" -ForegroundColor Red
                $failed++
            }
        } else {
            if ($response.note -eq 'mock') {
                Write-Host "  [PASS] $($test.Name) - Mock working!" -ForegroundColor Green
                Write-Host "        $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
                $passed++
            } elseif ($response.error -like "*not configured*") {
                Write-Host "  [FAIL] $($test.Name) - Still not working" -ForegroundColor Red
                Write-Host "        $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
                $failed++
            } else {
                Write-Host "  [????] $($test.Name) - Unexpected response" -ForegroundColor Yellow
                Write-Host "        $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
                $failed++
            }
        }
    } catch {
        Write-Host "  [ERROR] $($test.Name) - $_" -ForegroundColor Red
        $failed++
    }
}

# Summary
Write-Host "`n================================================" -ForegroundColor Cyan
if ($failed -eq 0) {
    Write-Host "       SUCCESS! All Mocks Working!             " -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The fix worked! Mocks are now responding correctly." -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run the full deployment:" -ForegroundColor Yellow
    Write-Host "  .\tools\release\Reset-And-Ship.ps1 -UsePM2" -ForegroundColor White
} else {
    Write-Host "      Some Tests Failed ($failed/$($passed+$failed))      " -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Check server logs:" -ForegroundColor Yellow
    Write-Host "  Receive-Job iris-fixed" -ForegroundColor White
}

Write-Host ""
Write-Host "Server is running. Commands:" -ForegroundColor Cyan
Write-Host "  Stop-Job iris-fixed       # Stop server" -ForegroundColor Gray
Write-Host "  Receive-Job iris-fixed    # View logs" -ForegroundColor Gray
Write-Host ""
Write-Host "Done." -ForegroundColor Green
