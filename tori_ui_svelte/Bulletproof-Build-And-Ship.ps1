# Bulletproof-Build-And-Ship.ps1
# Implements all hardenings from the review for 100% reliable mock deployment
# Includes: Cache-bust, PM2 env handling, fixed self-tests

param(
    [ValidateSet('mock','real')]
    [string]$Mode = 'mock',
    [int]$Port = 3000,
    [switch]$SkipTests,
    [switch]$UsePM2
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Set working directory
Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "    Bulletproof Build & Ship Pipeline v2.0     " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Mode: $Mode | Port: $Port" -ForegroundColor Yellow
Write-Host ""

# HARDENING 1: Cache-bust to force fresh compilation
Write-Host "[1/7] Cache-busting for fresh build..." -ForegroundColor Yellow

if (Test-Path .\.svelte-kit) { 
    Write-Host "  Removing .svelte-kit..." -ForegroundColor Gray
    Remove-Item .\.svelte-kit -Recurse -Force 
}
if (Test-Path .\node_modules\.vite) { 
    Write-Host "  Removing node_modules/.vite..." -ForegroundColor Gray
    Remove-Item .\node_modules\.vite -Recurse -Force 
}
Write-Host "  Cache cleared!" -ForegroundColor Green

# Step 2: Install dependencies
Write-Host "`n[2/7] Installing dependencies..." -ForegroundColor Yellow
& pnpm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
    exit 1
}
Write-Host "  Dependencies installed!" -ForegroundColor Green

# Step 3: Set environment based on mode
Write-Host "`n[3/7] Setting environment variables..." -ForegroundColor Yellow

if ($Mode -eq 'mock') {
    $env:IRIS_USE_MOCKS = "1"
    $env:IRIS_ALLOW_UNAUTH = "1"
    Write-Host "  IRIS_USE_MOCKS = 1" -ForegroundColor Green
    Write-Host "  IRIS_ALLOW_UNAUTH = 1" -ForegroundColor Green
} else {
    $env:IRIS_USE_MOCKS = "0"
    Write-Host "  IRIS_USE_MOCKS = 0 (real mode)" -ForegroundColor Yellow
}
$env:PORT = "$Port"
$env:NODE_ENV = "production"
Write-Host "  PORT = $Port" -ForegroundColor Green
Write-Host "  NODE_ENV = production" -ForegroundColor Green

# Step 4: Build
Write-Host "`n[4/7] Building application..." -ForegroundColor Yellow
& pnpm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Build successful!" -ForegroundColor Green

# Step 5: Verify mocks are in compiled output (if mock mode)
if ($Mode -eq 'mock') {
    Write-Host "`n[5/7] Verifying mocks in compiled output..." -ForegroundColor Yellow
    
    $compiledFiles = @(
        ".\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js",
        ".\.svelte-kit\output\server\entries\endpoints\api\memory\state\_server.ts.js"
    )
    
    $foundMocks = $false
    foreach ($file in $compiledFiles) {
        if (Test-Path $file) {
            $content = Get-Content $file -Raw
            if ($content -match 'note.*mock') {
                Write-Host "  Found mock in: $(Split-Path $file -Leaf)" -ForegroundColor Green
                $foundMocks = $true
            }
        }
    }
    
    if (-not $foundMocks) {
        Write-Host "  WARNING: Mocks not found in compiled output!" -ForegroundColor Red
        Write-Host "  The endpoints may not return mock data." -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[5/7] Skipping mock verification (real mode)" -ForegroundColor Gray
}

# Step 6: Launch the application
Write-Host "`n[6/7] Launching application..." -ForegroundColor Yellow

# Kill any existing process on the port
$existingProcess = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existingProcess) {
    $procId = $existingProcess.OwningProcess
    if ($procId -gt 0) {
        Write-Host "  Stopping existing process on port $Port (PID: $procId)..." -ForegroundColor Gray
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

if ($UsePM2) {
    # HARDENING 2: PM2 with proper environment handling
    Write-Host "  Starting with PM2..." -ForegroundColor Cyan
    
    # Check if PM2 is available (globally or locally)
    $pm2Exists = Get-Command pm2 -ErrorAction SilentlyContinue
    if (-not $pm2Exists) {
        Write-Host "  PM2 not found globally, checking local..." -ForegroundColor Yellow
        
        # Check if PM2 is in node_modules
        if (-not (Test-Path "node_modules\pm2")) {
            Write-Host "  Installing PM2 locally..." -ForegroundColor Yellow
            & npm install pm2
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Failed to install PM2" -ForegroundColor Red
                exit 1
            }
        }
        Write-Host "  Using npx to run PM2" -ForegroundColor Gray
    }
    
    # Stop and delete existing instance (use npx for safety)
    & npx pm2 stop iris-ui --silent 2>$null
    & npx pm2 delete iris-ui --silent 2>$null
    
    # Ensure PM2 gets the right environment
    if (-not $env:PORT) { $env:PORT = "$Port" }
    if (-not $env:IRIS_USE_MOCKS) { $env:IRIS_USE_MOCKS = $(if ($Mode -eq 'mock') { "1" } else { "0" }) }
    
    # Start with PM2 using --update-env to ensure fresh environment
    Write-Host "  PM2 starting with PORT=$env:PORT, IRIS_USE_MOCKS=$env:IRIS_USE_MOCKS" -ForegroundColor Gray
    & npx pm2 start .\build\index.js --name iris-ui --update-env --time --restart-delay 3000
    & npx pm2 save --silent
    
    Write-Host "  Application started with PM2!" -ForegroundColor Green
    Write-Host "  Commands:" -ForegroundColor Cyan
    Write-Host "    npx pm2 logs iris-ui       # View logs" -ForegroundColor Gray
    Write-Host "    npx pm2 restart iris-ui    # Restart" -ForegroundColor Gray
    Write-Host "    npx pm2 stop iris-ui       # Stop" -ForegroundColor Gray
    
} else {
    # Start with Node.js in background job
    Write-Host "  Starting with Node.js..." -ForegroundColor Cyan
    
    $job = Start-Job -Name "iris-server" -ScriptBlock {
        Set-Location "D:\Dev\kha\tori_ui_svelte"
        $env:PORT = $using:Port
        $env:IRIS_USE_MOCKS = $(if ($using:Mode -eq 'mock') { "1" } else { "0" })
        $env:IRIS_ALLOW_UNAUTH = $(if ($using:Mode -eq 'mock') { "1" } else { "0" })
        node build/index.js
    }
    
    Write-Host "  Application started in background job!" -ForegroundColor Green
    Write-Host "  Commands:" -ForegroundColor Cyan
    Write-Host "    Get-Job iris-server           # Check status" -ForegroundColor Gray
    Write-Host "    Receive-Job iris-server       # View logs" -ForegroundColor Gray
    Write-Host "    Stop-Job iris-server          # Stop" -ForegroundColor Gray
}

# Wait for server to start
Write-Host "`n  Waiting for server to start on port $Port..." -ForegroundColor Yellow
$maxWait = 20
$waited = 0
while ($waited -lt $maxWait) {
    $listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        Write-Host "  Server is listening!" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 1
    $waited++
    Write-Host "." -NoNewline
}
Write-Host ""

if ($waited -ge $maxWait) {
    Write-Host "ERROR: Server failed to start within $maxWait seconds!" -ForegroundColor Red
    if (-not $UsePM2) {
        Stop-Job -Name "iris-server" -ErrorAction SilentlyContinue
        Remove-Job -Name "iris-server" -ErrorAction SilentlyContinue
    }
    exit 1
}

# HARDENING 3: Clean self-test block
Write-Host "`n[7/7] Running smoke tests..." -ForegroundColor Yellow

function Test-Url {
    param([string]$Url, [int]$ExpectedStatus = 200)
    try {
        $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -Method GET -TimeoutSec 10
        return ($resp.StatusCode -eq $ExpectedStatus)
    } catch { 
        return $false 
    }
}

if (-not $SkipTests) {
    $tests = @(
        @{ Name = 'Health';   Url = "http://127.0.0.1:$Port/api/health";   Expect = 200 },
        @{ Name = 'Root';     Url = "http://127.0.0.1:$Port/";             Expect = 200 },
        @{ Name = 'Renderer'; Url = "http://127.0.0.1:$Port/renderer";     Expect = 200 },
        @{ Name = 'PDF API';  Url = "http://127.0.0.1:$Port/api/pdf/stats"; Expect = 200 },
        @{ Name = 'Memory API'; Url = "http://127.0.0.1:$Port/api/memory/state"; Expect = 200 }
    )
    
    $passed = 0
    $failed = 0
    
    foreach ($t in $tests) {
        if (Test-Url -Url $t.Url -ExpectedStatus $t.Expect) {
            Write-Host "  [PASS] $($t.Name)" -ForegroundColor Green
            $passed++
            
            # For API endpoints in mock mode, verify mock data
            if ($Mode -eq 'mock' -and $t.Url -like "*api*") {
                try {
                    $json = Invoke-RestMethod -Uri $t.Url -Method GET
                    if ($json.note -eq 'mock') {
                        Write-Host "        Mock data confirmed!" -ForegroundColor Cyan
                    }
                } catch {}
            }
        } else {
            Write-Host "  [FAIL] $($t.Name)" -ForegroundColor Red
            $failed++
        }
    }
    
    Write-Host "`n  Test Results: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
} else {
    Write-Host "  Tests skipped" -ForegroundColor Gray
}

# Final verification for mock mode
if ($Mode -eq 'mock') {
    Write-Host "`n[Verification] Testing mock endpoints..." -ForegroundColor Yellow
    
    try {
        $pdfStats = Invoke-RestMethod "http://127.0.0.1:$Port/api/pdf/stats"
        $memoryState = Invoke-RestMethod "http://127.0.0.1:$Port/api/memory/state"
        
        Write-Host "  PDF Stats:" -ForegroundColor Cyan
        Write-Host "    $($pdfStats | ConvertTo-Json -Compress)" -ForegroundColor Gray
        
        Write-Host "  Memory State:" -ForegroundColor Cyan
        Write-Host "    $($memoryState | ConvertTo-Json -Compress)" -ForegroundColor Gray
        
        if ($pdfStats.note -eq 'mock' -and $memoryState.note -eq 'mock') {
            Write-Host "`n  SUCCESS: Mock endpoints are working!" -ForegroundColor Green
        } else {
            Write-Host "`n  WARNING: Endpoints returned data but missing 'mock' note" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "  ERROR: Failed to verify mock endpoints - $_" -ForegroundColor Red
    }
}

Write-Host "`n================================================" -ForegroundColor Cyan
Write-Host "          Bulletproof Ship Complete!            " -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Application running at: http://127.0.0.1:$Port" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Yellow
Write-Host ""

if ($Mode -eq 'mock') {
    Write-Host "To switch to real mode:" -ForegroundColor Yellow
    Write-Host "  .\Bulletproof-Build-And-Ship.ps1 -Mode real -UsePM2" -ForegroundColor White
} else {
    Write-Host "To switch to mock mode:" -ForegroundColor Yellow
    Write-Host "  .\Bulletproof-Build-And-Ship.ps1 -Mode mock -UsePM2" -ForegroundColor White
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
