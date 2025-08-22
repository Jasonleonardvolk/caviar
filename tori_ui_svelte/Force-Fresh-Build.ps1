# Force-Fresh-Build.ps1
# Forces a completely fresh build with mocks compiled in

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "     Forcing Fresh Build with Mocks            " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: AGGRESSIVE cache clearing
Write-Host "[1] Aggressive cache clearing..." -ForegroundColor Yellow

# Stop any running servers first
Write-Host "  Stopping any running servers..." -ForegroundColor Gray
Stop-Job -Name "iris-*" -ErrorAction SilentlyContinue
& npx pm2 stop all --silent 2>$null
& npx pm2 delete all --silent 2>$null

# Kill any process on port 3000
$existingProcess = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
if ($existingProcess) {
    $procId = $existingProcess.OwningProcess
    if ($procId -gt 0) {
        Write-Host "  Killing process on port 3000 (PID: $procId)..." -ForegroundColor Gray
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

# Clear ALL build artifacts
$toDelete = @(
    ".\.svelte-kit",
    ".\build",
    ".\node_modules\.vite",
    ".\.vite",
    ".\dist",
    ".\.turbo"
)

foreach ($dir in $toDelete) {
    if (Test-Path $dir) {
        Write-Host "  Removing $dir..." -ForegroundColor Gray
        Remove-Item $dir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "  All caches cleared!" -ForegroundColor Green

# Step 2: Verify source files have mock gates
Write-Host "`n[2] Verifying source files..." -ForegroundColor Yellow

$filesToCheck = @(
    @{
        Path = "src\routes\api\pdf\stats\+server.ts"
        Name = "PDF stats endpoint"
    },
    @{
        Path = "src\routes\api\memory\state\+server.ts"
        Name = "Memory state endpoint"
    },
    @{
        Path = "src\routes\renderer\+page.server.ts"
        Name = "Renderer SSR"
    }
)

$allGood = $true
foreach ($file in $filesToCheck) {
    if (Test-Path $file.Path) {
        $content = Get-Content $file.Path -Raw
        if ($content -match "env\.IRIS_USE_MOCKS === '1'") {
            Write-Host "  $($file.Name): Has mock gate ✓" -ForegroundColor Green
        } else {
            Write-Host "  $($file.Name): MISSING mock gate ✗" -ForegroundColor Red
            $allGood = $false
        }
    } else {
        Write-Host "  $($file.Name): FILE NOT FOUND ✗" -ForegroundColor Red
        $allGood = $false
    }
}

if (-not $allGood) {
    Write-Host "`n  Some files missing mock gates. Fixing..." -ForegroundColor Yellow
    & .\Apply-All-Fixes.ps1 -SkipTests
}

# Step 3: Set environment for build
Write-Host "`n[3] Setting environment..." -ForegroundColor Yellow
$env:IRIS_USE_MOCKS = "1"
$env:PORT = "3000"
$env:NODE_ENV = "production"
$env:IRIS_ALLOW_UNAUTH = "1"

Write-Host "  IRIS_USE_MOCKS = 1" -ForegroundColor Green
Write-Host "  PORT = 3000" -ForegroundColor Green
Write-Host "  NODE_ENV = production" -ForegroundColor Green

# Step 4: Fresh install and build
Write-Host "`n[4] Installing dependencies..." -ForegroundColor Yellow
& pnpm install --force
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pnpm install failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Dependencies installed!" -ForegroundColor Green

Write-Host "`n[5] Building with mocks..." -ForegroundColor Yellow
& pnpm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed" -ForegroundColor Red
    exit 1
}
Write-Host "  Build complete!" -ForegroundColor Green

# Step 5: VERIFY mocks are in compiled output
Write-Host "`n[6] Verifying mocks in compiled output..." -ForegroundColor Yellow

$compiledFiles = @(
    ".\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js",
    ".\.svelte-kit\output\server\entries\endpoints\api\memory\state\_server.ts.js"
)

$mocksFound = $false
foreach ($file in $compiledFiles) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        if ($content -match 'IRIS_USE_MOCKS.*===.*["\']1["\']') {
            Write-Host "  Found mock check in: $(Split-Path $file -Leaf)" -ForegroundColor Green
            if ($content -match 'note.*:.*["\']mock["\']') {
                Write-Host "    Contains mock response ✓" -ForegroundColor Green
                $mocksFound = $true
            }
        } else {
            Write-Host "  NO MOCK CHECK in: $(Split-Path $file -Leaf)" -ForegroundColor Red
            # Show first 50 lines for debugging
            Write-Host "  First 50 lines of file:" -ForegroundColor Yellow
            $lines = $content -split "`n" | Select-Object -First 50
            $lines | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        }
    } else {
        Write-Host "  Compiled file not found: $file" -ForegroundColor Red
    }
}

if (-not $mocksFound) {
    Write-Host "`n  WARNING: Mocks may not be properly compiled!" -ForegroundColor Red
    Write-Host "  The server might still return 'not configured'" -ForegroundColor Yellow
}

# Step 6: Start server with Node.js
Write-Host "`n[7] Starting server with mocks..." -ForegroundColor Yellow

$job = Start-Job -Name "iris-mock-server" -ScriptBlock {
    Set-Location "D:\Dev\kha\tori_ui_svelte"
    $env:IRIS_USE_MOCKS = "1"
    $env:PORT = "3000"
    $env:IRIS_ALLOW_UNAUTH = "1"
    $env:NODE_ENV = "production"
    Write-Host "Starting with IRIS_USE_MOCKS=$env:IRIS_USE_MOCKS"
    node build/index.js
}

# Wait for server
Write-Host "  Waiting for server to start..."
$maxWait = 20
$waited = 0
while ($waited -lt $maxWait) {
    $listening = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
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
    Write-Host "ERROR: Server failed to start!" -ForegroundColor Red
    Stop-Job -Name "iris-mock-server" -ErrorAction SilentlyContinue
    Remove-Job -Name "iris-mock-server" -ErrorAction SilentlyContinue
    exit 1
}

# Step 7: Test the endpoints
Write-Host "`n[8] Testing mock endpoints..." -ForegroundColor Yellow

Start-Sleep -Seconds 2  # Give server time to fully initialize

$endpoints = @(
    @{ Name = "PDF Stats"; Url = "http://127.0.0.1:3000/api/pdf/stats" },
    @{ Name = "Memory State"; Url = "http://127.0.0.1:3000/api/memory/state" }
)

$allWork = $true
foreach ($endpoint in $endpoints) {
    try {
        Write-Host "  Testing $($endpoint.Name)..." -ForegroundColor Gray
        $response = Invoke-RestMethod -Uri $endpoint.Url -Method GET
        
        if ($response.note -eq 'mock') {
            Write-Host "    SUCCESS: Mock data returned!" -ForegroundColor Green
            Write-Host "    Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
        } elseif ($response.error -like "*not configured*") {
            Write-Host "    FAILED: Still returning 'not configured'" -ForegroundColor Red
            Write-Host "    Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
            $allWork = $false
        } else {
            Write-Host "    WARNING: Unexpected response" -ForegroundColor Yellow
            Write-Host "    Response: $($response | ConvertTo-Json -Compress)" -ForegroundColor DarkGray
        }
    } catch {
        Write-Host "    ERROR: $_" -ForegroundColor Red
        $allWork = $false
    }
}

# Summary
Write-Host "`n================================================" -ForegroundColor Cyan
if ($allWork) {
    Write-Host "         SUCCESS! Mocks are working!           " -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Server running at: http://127.0.0.1:3000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  Get-Job iris-mock-server      # Check status" -ForegroundColor Gray
    Write-Host "  Receive-Job iris-mock-server  # View logs" -ForegroundColor Gray
    Write-Host "  Stop-Job iris-mock-server     # Stop server" -ForegroundColor Gray
} else {
    Write-Host "      PROBLEM: Mocks still not working!        " -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "1. Check server logs:" -ForegroundColor Gray
    Write-Host "   Receive-Job iris-mock-server" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Check environment in running process:" -ForegroundColor Gray
    Write-Host "   Get-Job iris-mock-server | Receive-Job -Keep" -ForegroundColor White
    Write-Host ""
    Write-Host "3. Try manual rebuild:" -ForegroundColor Gray
    Write-Host "   Stop-Job iris-mock-server" -ForegroundColor White
    Write-Host "   Remove-Item .\.svelte-kit -Recurse -Force" -ForegroundColor White
    Write-Host "   pnpm run build" -ForegroundColor White
    Write-Host ""
    Write-Host "Server is still running for debugging." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
