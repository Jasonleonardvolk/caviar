# Apply-All-Fixes.ps1
# Applies all fixes from the PUSH.txt transcript analysis
# Fixes: Mock endpoints, SSR errors, and verifies everything works

param(
    [switch]$SkipBuild,
    [switch]$SkipTests
)

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "     Applying All Fixes from PUSH.txt      " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Fix A: Verify mocks are compiled and force them if not
Write-Host "[Fix A] Checking if mocks are compiled..." -ForegroundColor Yellow

# Check if compiled endpoints contain mock text
$compiledPdfStats = ".\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js"
$compiledMemoryState = ".\.svelte-kit\output\server\entries\endpoints\api\memory\state\_server.ts.js"

$needsRebuild = $false

if ((Test-Path $compiledPdfStats) -and (Test-Path $compiledMemoryState)) {
    $pdfContent = Get-Content $compiledPdfStats -Raw
    $memoryContent = Get-Content $compiledMemoryState -Raw
    
    if ($pdfContent -match 'note.*mock' -and $memoryContent -match 'note.*mock') {
        Write-Host "  Mocks are already compiled!" -ForegroundColor Green
    } else {
        Write-Host "  Mocks NOT found in compiled output!" -ForegroundColor Yellow
        $needsRebuild = $true
    }
} else {
    Write-Host "  Compiled files don't exist yet" -ForegroundColor Yellow
    $needsRebuild = $true
}

# Ensure source files have the correct mock gates
Write-Host "`n  Verifying source files have mock gates..." -ForegroundColor Cyan

# Check PDF stats endpoint
$pdfSource = Get-Content "src\routes\api\pdf\stats\+server.ts" -Raw
if ($pdfSource -notmatch "env.IRIS_USE_MOCKS === '1'") {
    Write-Host "  Updating PDF stats endpoint..." -ForegroundColor Yellow
    @'
import { json } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';

export async function GET() {
  if (env.IRIS_USE_MOCKS === '1') {
    return json({ ok: true, docs: 3, pages: 42, note: 'mock' });
  }
  
  // Real service implementation would go here
  return json({
    ok: false,
    error: 'PDF processing service not configured',
    hint: 'Set IRIS_USE_MOCKS=1 to use mock data'
  }, { status: 503 });
}
'@ | Out-File -FilePath "src\routes\api\pdf\stats\+server.ts" -Encoding UTF8
    $needsRebuild = $true
}

# Check Memory state endpoint
$memorySource = Get-Content "src\routes\api\memory\state\+server.ts" -Raw
if ($memorySource -notmatch "env.IRIS_USE_MOCKS === '1'") {
    Write-Host "  Updating Memory state endpoint..." -ForegroundColor Yellow
    @'
import { json } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';

export async function GET() {
  if (env.IRIS_USE_MOCKS === '1') {
    return json({
      ok: true,
      state: 'idle',
      concepts: 0,
      vault: { connected: false, mode: 'mock' },
      note: 'mock'
    });
  }
  
  // Real service implementation would go here
  return json({
    ok: false,
    error: 'Memory vault service not configured',
    hint: 'Set IRIS_USE_MOCKS=1 to use mock data'
  }, { status: 503 });
}
'@ | Out-File -FilePath "src\routes\api\memory\state\+server.ts" -Encoding UTF8
    $needsRebuild = $true
}

Write-Host "  Source files verified!" -ForegroundColor Green

# Fix C: Add renderer guard for SSR
Write-Host "`n[Fix C] Adding renderer SSR guard..." -ForegroundColor Yellow

if (!(Test-Path "src\routes\renderer\+page.server.ts")) {
    Write-Host "  Creating renderer +page.server.ts..." -ForegroundColor Cyan
    @'
import type { PageServerLoad } from './$types';
import { env } from '$env/dynamic/private';

export const load: PageServerLoad = async ({ fetch }) => {
  // If using mocks, return mock data directly without fetching
  if (env.IRIS_USE_MOCKS === '1') {
    return {
      pdf: { ok: true, docs: 3, pages: 42, note: 'mock' },
      memory: { ok: true, state: 'idle', concepts: 0, vault: { connected: false, mode: 'mock' }, note: 'mock' }
    };
  }
  
  // In real mode, try to fetch from the API endpoints
  try {
    const [pdfResponse, memoryResponse] = await Promise.all([
      fetch('/api/pdf/stats'),
      fetch('/api/memory/state')
    ]);
    
    const pdf = await pdfResponse.json();
    const memory = await memoryResponse.json();
    
    return {
      pdf,
      memory
    };
  } catch (error) {
    // If there's an error in real mode, return safe defaults
    console.error('Error fetching data for renderer:', error);
    return {
      pdf: { ok: false, error: 'Unable to fetch PDF stats' },
      memory: { ok: false, error: 'Unable to fetch memory state' }
    };
  }
};
'@ | Out-File -FilePath "src\routes\renderer\+page.server.ts" -Encoding UTF8
    $needsRebuild = $true
    Write-Host "  Created renderer SSR handler!" -ForegroundColor Green
} else {
    Write-Host "  Renderer SSR handler already exists!" -ForegroundColor Green
}

# Rebuild if needed
if ($needsRebuild -and !$SkipBuild) {
    Write-Host "`n[Build] Rebuilding application..." -ForegroundColor Yellow
    & pnpm run build
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "Build successful!" -ForegroundColor Green
    
    # Verify mocks are now in compiled output
    Write-Host "`nVerifying mocks in compiled output..." -ForegroundColor Cyan
    $result = Select-String -Path $compiledPdfStats, $compiledMemoryState -Pattern "note.*mock" 2>$null
    if ($result) {
        Write-Host "  SUCCESS: Mocks found in compiled output!" -ForegroundColor Green
        $result | ForEach-Object { Write-Host "    Found in: $($_.Filename)" -ForegroundColor Gray }
    } else {
        Write-Host "  WARNING: Mocks still not in compiled output" -ForegroundColor Yellow
    }
} elseif ($SkipBuild) {
    Write-Host "`n[Build] Skipping build (use existing)" -ForegroundColor Yellow
}

# Run the test with the fixes applied
if (!$SkipTests) {
    Write-Host "`n[Test] Running verification with mocks..." -ForegroundColor Yellow
    
    # Set environment for mocks
    $env:IRIS_USE_MOCKS = "1"
    $env:PORT = "3000"
    $env:IRIS_ALLOW_UNAUTH = "1"
    
    # Run the verification script
    & .\tools\release\Verify-EndToEnd.ps1 -Mode mock -Port 3000 -StartServer -StopOnExit
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n============================================" -ForegroundColor Green
        Write-Host "   ALL FIXES APPLIED SUCCESSFULLY!         " -ForegroundColor Green
        Write-Host "============================================" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run the full deployment:" -ForegroundColor Cyan
        Write-Host "  .\tools\release\Reset-And-Ship.ps1 -UsePM2" -ForegroundColor White
        Write-Host ""
        Write-Host "Or test manually:" -ForegroundColor Cyan
        Write-Host "  Invoke-RestMethod http://127.0.0.1:3000/api/pdf/stats" -ForegroundColor White
        Write-Host "  Invoke-RestMethod http://127.0.0.1:3000/api/memory/state" -ForegroundColor White
    } else {
        Write-Host "`nSome tests failed. Check the output above." -ForegroundColor Red
    }
} else {
    Write-Host "`n[Test] Skipping tests" -ForegroundColor Yellow
    Write-Host "`nFixes applied! Run this to test:" -ForegroundColor Cyan
    Write-Host "  .\tools\release\Verify-EndToEnd.ps1 -Mode mock -Port 3000 -StartServer -StopOnExit" -ForegroundColor White
}
