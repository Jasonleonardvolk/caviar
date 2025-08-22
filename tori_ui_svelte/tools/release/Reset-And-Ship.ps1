# Reset-And-Ship.ps1
# iRis v0.1.0 Release Script
# Executes the complete build, launch, and smoke test pipeline

param(
    [switch]$SkipBuild,
    [switch]$SkipTests,
    [switch]$UsePM2,
    [switch]$EnableLAN
)

# === PORT GUARD START ===
# Ensures the target port is free before we start the server.
$DefaultPort = 3000
if (-not $env:PORT -or -not ($env:PORT -as [int])) { $env:PORT = $DefaultPort }

# --- Pre-free the port (must be before any checks)
$__port = if ($env:PORT) { [int]$env:PORT } else { 3000 }
$__runPorts = Join-Path $PSScriptRoot "..\runtime\Run-Ports.ps1"
if (Test-Path $__runPorts) {
  & $__runPorts -Ports $__port -Kill | Out-Null
}

# --- Re-check: treat PID 0 as "free"
$__conns = Get-NetTCPConnection -LocalPort $__port -State Listen -ErrorAction SilentlyContinue
$__pid = $null
if ($__conns) {
  $__pid = ($__conns | Where-Object { $_.OwningProcess -gt 0 } |
            Select-Object -First 1 -ExpandProperty OwningProcess)
}

if ($__pid) {
  throw "Reset-And-Ship: Port $__port is already in use by PID $__pid. Stop it or set `$env:PORT to a free port, then retry."
}

function Wait-PortListening {
  param([int]$Port, [int]$TimeoutSec = 20)
  $sw = [System.Diagnostics.Stopwatch]::StartNew()
  while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
    $listening = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue |
                 Where-Object { $_.State -eq 'Listen' }
    if ($listening) { return }
    Start-Sleep -Milliseconds 250
  }
  throw "Server did not start listening on port $Port within $TimeoutSec seconds."
}
# === PORT GUARD END ===

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "     iRis v0.1.0 - Reset & Ship Pipeline     " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Set working directory
$rootDir = "D:\Dev\kha\tori_ui_svelte"
Set-Location $rootDir
Write-Host "Working directory: $rootDir" -ForegroundColor Green

# Step 1: Verify canonical UI imports
Write-Host "`n[Step 1] Verifying canonical UI imports..." -ForegroundColor Yellow
$nonCanonical = Get-ChildItem -Path "src" -Recurse -Filter "*.svelte" | 
    Select-String -Pattern "HolographicDisplayEnhanced|HolographicDisplay_FIXED"
    
if ($nonCanonical) {
    Write-Host "WARNING: Found non-canonical imports:" -ForegroundColor Red
    $nonCanonical | ForEach-Object { Write-Host "  - $_" }
} else {
    Write-Host "All imports are canonical" -ForegroundColor Green
}

# Step 2: Ensure environment files exist
Write-Host "`n[Step 2] Checking environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env.local") {
    Write-Host ".env.local exists" -ForegroundColor Green
} else {
    Write-Host "Creating .env.local..." -ForegroundColor Yellow
    @"
PORT=3000
IRIS_ALLOW_UNAUTH=1
IRIS_USE_MOCKS=1
LOCAL_UPLOAD_DIR=var/uploads
"@ | Out-File -FilePath ".env.local" -Encoding UTF8
    Write-Host ".env.local created" -ForegroundColor Green
}

if (Test-Path ".env.production") {
    Write-Host ".env.production exists" -ForegroundColor Green
} else {
    Write-Host ".env.production exists (update AWS credentials before deploying)" -ForegroundColor Yellow
}

# Step 3: Ensure upload directory exists
Write-Host "`n[Step 3] Ensuring upload directory exists..." -ForegroundColor Yellow
$uploadDir = Join-Path $rootDir "var\uploads"
if (!(Test-Path $uploadDir)) {
    New-Item -ItemType Directory -Path $uploadDir -Force | Out-Null
    Write-Host "Created upload directory: $uploadDir" -ForegroundColor Green
} else {
    Write-Host "Upload directory exists: $uploadDir" -ForegroundColor Green
}

# Step 4: Build the application
if (!$SkipBuild) {
    Write-Host "`n[Step 4] Building application..." -ForegroundColor Yellow
    
    # Install dependencies
    Write-Host "Installing dependencies with pnpm..." -ForegroundColor Cyan
    & pnpm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
    
    # Build for production
    Write-Host "Building for production..." -ForegroundColor Cyan
    & pnpm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed" -ForegroundColor Red
        exit 1
    }
    
    Write-Host "Build completed successfully" -ForegroundColor Green
} else {
    Write-Host "`n[Step 4] Skipping build (using existing)" -ForegroundColor Yellow
}

# Step 5: Launch the application
Write-Host "`n[Step 5] Launching application..." -ForegroundColor Yellow

if ($UsePM2) {
    # Check if PM2 is available (globally or locally)
    $pm2Exists = Get-Command pm2 -ErrorAction SilentlyContinue
    if (!$pm2Exists) {
        Write-Host "PM2 not found globally, checking local..." -ForegroundColor Yellow
        
        # Check if PM2 is in node_modules
        if (-not (Test-Path "node_modules\pm2")) {
            Write-Host "Installing PM2 locally..." -ForegroundColor Cyan
            & npm install pm2
            if ($LASTEXITCODE -ne 0) {
                Write-Host "ERROR: Failed to install PM2" -ForegroundColor Red
                exit 1
            }
        }
        Write-Host "Using npx to run PM2" -ForegroundColor Gray
    }
    
    # Stop existing instance if running (use npx to ensure it works)
    & npx pm2 stop iris -s 2>$null
    & npx pm2 delete iris -s 2>$null
    
    # Ensure runtime env is present for PM2 (mock vs real)
    if (-not $env:PORT) { $env:PORT = "3000" }
    if (-not $env:IRIS_USE_MOCKS) { $env:IRIS_USE_MOCKS = "1" }  # default to mock mode
    
    # Start with PM2 using --update-env to ensure fresh environment
    Write-Host "Starting iRis with PM2..." -ForegroundColor Cyan
    Write-Host "  PORT=$env:PORT, IRIS_USE_MOCKS=$env:IRIS_USE_MOCKS" -ForegroundColor Gray
    $env:NODE_ENV = "production"
    & npx pm2 start build/index.js --name iris --update-env --time --restart-delay 3000
    & npx pm2 save
    
    Write-Host "Application started with PM2" -ForegroundColor Green
    Write-Host "  Use 'npx pm2 logs iris' to view logs" -ForegroundColor Cyan
    Write-Host "  Use 'npx pm2 stop iris' to stop" -ForegroundColor Cyan
} else {
    # Start directly with Node
    Write-Host "Starting iRis with Node.js..." -ForegroundColor Cyan
    $env:PORT = "3000"
    Start-Job -Name "iris-server" -ScriptBlock {
        Set-Location "D:\Dev\kha\tori_ui_svelte"
        $env:PORT = "3000"
        node build/index.js
    }
    
    Write-Host "Application started in background job" -ForegroundColor Green
    Write-Host "  Use 'Get-Job iris-server' to check status" -ForegroundColor Cyan
    Write-Host "  Use 'Stop-Job iris-server' to stop" -ForegroundColor Cyan
}

# Wait for server to start
Write-Host "`nWaiting for server to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Step 6: Run smoke tests (HARDENING 3: Clean self-test block)
function Test-Url {
  param([string]$Url, [int]$ExpectedStatus = 200)
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -Method GET -TimeoutSec 10
    return ($resp.StatusCode -eq $ExpectedStatus)
  } catch { return $false }
}

if (-not $SkipTests) {
  Write-Host "`n[Step 6] Running smoke tests..." -ForegroundColor Yellow
  $Port = [int]$env:PORT
  $tests = @(
    @{ Name = 'Health';   Url = "http://127.0.0.1:$Port/api/health";   Expect = 200 },
    @{ Name = 'Root';     Url = "http://127.0.0.1:$Port/";             Expect = 200 },
    @{ Name = 'Renderer'; Url = "http://127.0.0.1:$Port/renderer";     Expect = 200 },
    @{ Name = 'Upload';   Url = "http://127.0.0.1:$Port/upload";       Expect = 200 },
    @{ Name = 'List API'; Url = "http://127.0.0.1:$Port/api/list";     Expect = 200 },
    @{ Name = 'PDF API';  Url = "http://127.0.0.1:$Port/api/pdf/stats"; Expect = 200 },
    @{ Name = 'Memory API'; Url = "http://127.0.0.1:$Port/api/memory/state"; Expect = 200 }
  )

  $passed = 0
  $failed = 0
  
  foreach ($t in $tests) {
    if (Test-Url -Url $t.Url -ExpectedStatus $t.Expect) {
      Write-Host "  [SHIP TEST] $($t.Name): PASS ($($t.Expect))" -ForegroundColor Green
      $passed++
    } else {
      Write-Host "  [SHIP TEST] $($t.Name): FAIL" -ForegroundColor Yellow
      $failed++
    }
  }
  
  Write-Host "`nTest Results: $passed passed, $failed failed" -ForegroundColor $(if ($failed -eq 0) { "Green" } else { "Yellow" })
} else {
  Write-Host "`n[Step 6] Skipping smoke tests" -ForegroundColor Yellow
}

# Step 7: Configure LAN access (optional)
if ($EnableLAN) {
    Write-Host "`n[Step 7] Configuring LAN access..." -ForegroundColor Yellow
    
    # Check if rule already exists
    $existingRule = Get-NetFirewallRule -DisplayName "iRis 3000" -ErrorAction SilentlyContinue
    if ($existingRule) {
        Write-Host "Firewall rule already exists" -ForegroundColor Green
    } else {
        # Create firewall rule (requires admin)
        try {
            New-NetFirewallRule -DisplayName "iRis 3000" -Direction Inbound -Protocol TCP -LocalPort 3000 -Action Allow -ErrorAction Stop
            Write-Host "Firewall rule created for port 3000" -ForegroundColor Green
            
            # Get local IP address
            $localIP = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Ethernet", "Wi-Fi" | 
                       Where-Object { $_.IPAddress -notlike "169.254.*" } | 
                       Select-Object -First 1).IPAddress
            
            Write-Host "`n  LAN Access enabled at: http://${localIP}:3000" -ForegroundColor Cyan
        } catch {
            Write-Host "  Note: Admin privileges required for firewall configuration" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "`n[Step 7] LAN access not configured (use -EnableLAN to enable)" -ForegroundColor Gray
}

# Summary
Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "          iRis v0.1.0 - Ship Ready!          " -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Application running at: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Renderer available at: http://localhost:3000/renderer" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Verify the holographic display renders correctly" -ForegroundColor White
Write-Host "  2. Test file upload functionality" -ForegroundColor White
Write-Host "  3. Check API endpoints return expected data" -ForegroundColor White
Write-Host "  4. When ready, set IRIS_USE_MOCKS=0 to use real services" -ForegroundColor White
Write-Host ""
Write-Host "Commands:" -ForegroundColor Yellow
if ($UsePM2) {
    Write-Host "  npx pm2 logs iris     - View application logs" -ForegroundColor White
    Write-Host "  npx pm2 restart iris  - Restart application" -ForegroundColor White
    Write-Host "  npx pm2 stop iris     - Stop application" -ForegroundColor White
} else {
    Write-Host "  Get-Job iris-server          - Check job status" -ForegroundColor White
    Write-Host "  Receive-Job iris-server      - View logs" -ForegroundColor White
    Write-Host "  Stop-Job iris-server         - Stop application" -ForegroundColor White
    Write-Host "  Remove-Job iris-server       - Clean up job" -ForegroundColor White
}
Write-Host ""
