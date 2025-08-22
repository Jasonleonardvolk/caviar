# Fix-Dependencies-And-Mocks.ps1
# Fixes missing dependencies and implements centralized mock detection

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "    Fixing Dependencies & Mock Detection       " -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install missing adapter-node
Write-Host "[1] Installing missing dependencies..." -ForegroundColor Yellow
Write-Host "  Installing @sveltejs/adapter-node..." -ForegroundColor Gray

& pnpm add -D @sveltejs/adapter-node
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Trying with npm..." -ForegroundColor Yellow
    & npm install --save-dev @sveltejs/adapter-node
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install adapter-node!" -ForegroundColor Red
        Write-Host "Try manually: pnpm add -D @sveltejs/adapter-node" -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "  adapter-node installed!" -ForegroundColor Green

# Step 2: Ensure all dependencies are installed
Write-Host "`n[2] Installing all dependencies..." -ForegroundColor Yellow
& pnpm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies!" -ForegroundColor Red
    exit 1
}
Write-Host "  Dependencies installed!" -ForegroundColor Green

# Step 3: Create centralized env helper
Write-Host "`n[3] Creating centralized mock detection..." -ForegroundColor Yellow

# Create lib/server directory if it doesn't exist
$libServerDir = "src\lib\server"
if (!(Test-Path $libServerDir)) {
    New-Item -ItemType Directory -Path $libServerDir -Force | Out-Null
    Write-Host "  Created $libServerDir" -ForegroundColor Gray
}

# Create env.ts with centralized mock detection
@'
// src/lib/server/env.ts
// Centralized environment detection for mock mode
import { env as dyn } from '$env/dynamic/private';

// Treat 1/true/yes (any case) as truthy
const truthy = (v: unknown): boolean =>
  typeof v === 'string' && /^(1|true|yes|y)$/i.test(v);

// IRIS_FORCE_MOCKS overrides IRIS_USE_MOCKS (both checked from Kit + Node)
export const isMock = (): boolean => {
  const force = dyn.IRIS_FORCE_MOCKS ?? process.env.IRIS_FORCE_MOCKS;
  const use   = dyn.IRIS_USE_MOCKS   ?? process.env.IRIS_USE_MOCKS;
  return truthy(force) || truthy(use);
};

// Export for debugging
export const getMockStatus = () => ({
  force: dyn.IRIS_FORCE_MOCKS ?? process.env.IRIS_FORCE_MOCKS,
  use: dyn.IRIS_USE_MOCKS ?? process.env.IRIS_USE_MOCKS,
  isMock: isMock()
});
'@ | Out-File -FilePath "$libServerDir\env.ts" -Encoding UTF8

Write-Host "  Created centralized env.ts" -ForegroundColor Green

# Step 4: Update PDF endpoint
Write-Host "`n[4] Updating endpoints to use centralized detection..." -ForegroundColor Yellow

@'
import { json } from '@sveltejs/kit';
import { isMock } from '$lib/server/env';

export async function GET() {
  if (isMock()) {
    return json({ ok: true, docs: 3, pages: 42, note: 'mock' });
  }
  
  // Real path - TODO: Connect to real PDF processing service
  return json({
    ok: false,
    error: 'PDF processing service not configured',
    hint: 'Set IRIS_USE_MOCKS=1 to use mock data'
  }, { status: 503 });
}
'@ | Out-File -FilePath "src\routes\api\pdf\stats\+server.ts" -Encoding UTF8

Write-Host "  Updated PDF endpoint" -ForegroundColor Green

# Step 5: Update Memory endpoint
@'
import { json } from '@sveltejs/kit';
import { isMock } from '$lib/server/env';

export async function GET() {
  if (isMock()) {
    return json({
      ok: true,
      state: 'idle',
      concepts: 0,
      vault: { connected: false, mode: 'mock' },
      note: 'mock'
    });
  }
  
  // Real path - TODO: Connect to real memory vault service
  return json({
    ok: false,
    error: 'Memory vault service not configured',
    hint: 'Set IRIS_USE_MOCKS=1 to use mock data'
  }, { status: 503 });
}
'@ | Out-File -FilePath "src\routes\api\memory\state\+server.ts" -Encoding UTF8

Write-Host "  Updated Memory endpoint" -ForegroundColor Green

# Step 6: Create/Update Health endpoint
Write-Host "`n[5] Creating health endpoint..." -ForegroundColor Yellow

$healthDir = "src\routes\api\health"
if (!(Test-Path $healthDir)) {
    New-Item -ItemType Directory -Path $healthDir -Force | Out-Null
}

@'
import { json } from '@sveltejs/kit';
import { isMock, getMockStatus } from '$lib/server/env';

export async function GET() {
  return json({ 
    ok: true, 
    status: 'ok',
    mode: isMock() ? 'mock' : 'real',
    debug: getMockStatus()
  });
}
'@ | Out-File -FilePath "$healthDir\+server.ts" -Encoding UTF8

Write-Host "  Created health endpoint" -ForegroundColor Green

# Step 7: Clear cache and build
Write-Host "`n[6] Building application..." -ForegroundColor Yellow

# Clear cache
if (Test-Path ".\.svelte-kit") {
    Remove-Item ".\.svelte-kit" -Recurse -Force
    Write-Host "  Cleared .svelte-kit cache" -ForegroundColor Gray
}

# Set environment for build
$env:IRIS_USE_MOCKS = "1"
$env:PORT = "3000"

# Build
Write-Host "  Building..." -ForegroundColor Gray
& pnpm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Build failed!" -ForegroundColor Red
    Write-Host "Check if all dependencies are installed:" -ForegroundColor Yellow
    Write-Host "  pnpm install" -ForegroundColor White
    exit 1
}
Write-Host "  Build successful!" -ForegroundColor Green

# Step 8: Quick test
Write-Host "`n[7] Starting quick test..." -ForegroundColor Yellow

# Start server
$job = Start-Job -Name "iris-test" -ScriptBlock {
    Set-Location "D:\Dev\kha\tori_ui_svelte"
    $env:IRIS_USE_MOCKS = "1"
    $env:PORT = "3000"
    node build/index.js
}

# Wait for server
Write-Host "  Waiting for server..."
$waited = 0
while ($waited -lt 20) {
    $tcp = Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue
    if ($tcp) {
        Write-Host "  Server is listening!" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 1
    $waited++
}

if ($waited -ge 20) {
    Write-Host "  Server didn't start - continuing anyway" -ForegroundColor Yellow
}

Start-Sleep -Seconds 2

# Test endpoints
Write-Host "`n[8] Testing endpoints..." -ForegroundColor Yellow

$tests = @(
    @{ Name = "Health"; Url = "http://127.0.0.1:3000/api/health" },
    @{ Name = "PDF Stats"; Url = "http://127.0.0.1:3000/api/pdf/stats" },
    @{ Name = "Memory State"; Url = "http://127.0.0.1:3000/api/memory/state" }
)

$allPass = $true
foreach ($test in $tests) {
    try {
        $response = Invoke-RestMethod -Uri $test.Url -Method GET
        if ($response.ok -or $response.status -eq "ok") {
            if ($response.note -eq "mock" -or $response.mode -eq "mock") {
                Write-Host "  [PASS] $($test.Name) - Mock working!" -ForegroundColor Green
            } else {
                Write-Host "  [PASS] $($test.Name)" -ForegroundColor Green
            }
        } else {
            Write-Host "  [FAIL] $($test.Name)" -ForegroundColor Red
            $allPass = $false
        }
    } catch {
        Write-Host "  [ERROR] $($test.Name) - $_" -ForegroundColor Red
        $allPass = $false
    }
}

# Stop test server
Stop-Job -Name "iris-test" -ErrorAction SilentlyContinue
Remove-Job -Name "iris-test" -ErrorAction SilentlyContinue

# Summary
Write-Host "`n================================================" -ForegroundColor Cyan
if ($allPass) {
    Write-Host "        SUCCESS! Everything Fixed!             " -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Dependencies installed and mocks working!" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run:" -ForegroundColor Yellow
    Write-Host "  .\tools\release\Reset-And-Ship.ps1 -UsePM2" -ForegroundColor White
} else {
    Write-Host "         Some Tests Failed                     " -ForegroundColor Red  
    Write-Host "================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "But dependencies are fixed! Try:" -ForegroundColor Yellow
    Write-Host "  node build/index.js" -ForegroundColor White
    Write-Host "Then test manually:" -ForegroundColor Gray
    Write-Host "  Invoke-RestMethod http://127.0.0.1:3000/api/health" -ForegroundColor White
}

Write-Host "`nDone." -ForegroundColor Green
