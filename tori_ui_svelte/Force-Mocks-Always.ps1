# Force-Mocks-Always.ps1
# Nuclear option - hardcodes endpoints to ALWAYS return mocks
# Use this if environment detection still doesn't work

param(
    [switch]$Revert  # Revert to environment-based mocks
)

Set-Location "D:\Dev\kha\tori_ui_svelte"

Write-Host "================================================" -ForegroundColor Cyan
if ($Revert) {
    Write-Host "      Reverting to Environment-Based Mocks     " -ForegroundColor Cyan
} else {
    Write-Host "      FORCING MOCKS ALWAYS (Nuclear Option)    " -ForegroundColor Red
}
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

if ($Revert) {
    # Revert to environment-based mocks
    Write-Host "Reverting endpoints to check environment..." -ForegroundColor Yellow
    
    # PDF endpoint
    @'
import { json } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';

export async function GET() {
  // Check multiple sources for IRIS_USE_MOCKS
  const useMocks = env.IRIS_USE_MOCKS === '1' || 
                   env.IRIS_USE_MOCKS === 'true' ||
                   process.env.IRIS_USE_MOCKS === '1' ||
                   process.env.IRIS_USE_MOCKS === 'true';
  
  if (useMocks) {
    return json({ ok: true, docs: 3, pages: 42, note: 'mock' });
  }
  
  // TODO: Connect to real PDF processing service
  // For now, return a placeholder indicating real service is not available
  return json({
    ok: false,
    error: 'PDF processing service not configured',
    hint: 'Set IRIS_USE_MOCKS=1 to use mock data'
  }, { status: 503 });
}
'@ | Out-File -FilePath "src\routes\api\pdf\stats\+server.ts" -Encoding UTF8
    
    # Memory endpoint
    @'
import { json } from '@sveltejs/kit';
import { env } from '$env/dynamic/private';

export async function GET() {
  // Check multiple sources for IRIS_USE_MOCKS
  const useMocks = env.IRIS_USE_MOCKS === '1' || 
                   env.IRIS_USE_MOCKS === 'true' ||
                   process.env.IRIS_USE_MOCKS === '1' ||
                   process.env.IRIS_USE_MOCKS === 'true';
  
  if (useMocks) {
    return json({
      ok: true,
      state: 'idle',
      concepts: 0,
      vault: { connected: false, mode: 'mock' },
      note: 'mock'
    });
  }
  
  // TODO: Connect to real memory vault service
  // For now, return a placeholder indicating real service is not available
  return json({
    ok: false,
    error: 'Memory vault service not configured',
    hint: 'Set IRIS_USE_MOCKS=1 to use mock data'
  }, { status: 503 });
}
'@ | Out-File -FilePath "src\routes\api\memory\state\+server.ts" -Encoding UTF8
    
    Write-Host "Endpoints reverted to environment-based mocks" -ForegroundColor Green
    
} else {
    # Force mocks always
    Write-Host "WARNING: This will make endpoints ALWAYS return mocks!" -ForegroundColor Red
    Write-Host "Use -Revert flag to undo this change later." -ForegroundColor Yellow
    Write-Host ""
    
    $confirm = Read-Host "Are you sure? (yes/no)"
    if ($confirm -ne "yes") {
        Write-Host "Cancelled." -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host "`nForcing endpoints to always return mocks..." -ForegroundColor Yellow
    
    # PDF endpoint - ALWAYS returns mock
    @'
import { json } from '@sveltejs/kit';

export async function GET() {
  // FORCED MOCK - Always returns mock data regardless of environment
  return json({ ok: true, docs: 3, pages: 42, note: 'mock' });
}
'@ | Out-File -FilePath "src\routes\api\pdf\stats\+server.ts" -Encoding UTF8
    
    # Memory endpoint - ALWAYS returns mock
    @'
import { json } from '@sveltejs/kit';

export async function GET() {
  // FORCED MOCK - Always returns mock data regardless of environment
  return json({
    ok: true,
    state: 'idle',
    concepts: 0,
    vault: { connected: false, mode: 'mock' },
    note: 'mock'
  });
}
'@ | Out-File -FilePath "src\routes\api\memory\state\+server.ts" -Encoding UTF8
    
    Write-Host "Endpoints now ALWAYS return mocks!" -ForegroundColor Green
}

# Rebuild
Write-Host "`nRebuilding..." -ForegroundColor Yellow

# Clear cache
if (Test-Path ".\.svelte-kit") {
    Remove-Item ".\.svelte-kit" -Recurse -Force
}

& pnpm run build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`nBuild complete!" -ForegroundColor Green

if (-not $Revert) {
    Write-Host "`n================================================" -ForegroundColor Red
    Write-Host "     ENDPOINTS NOW ALWAYS RETURN MOCKS!        " -ForegroundColor Red
    Write-Host "================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "This is a temporary fix for testing only!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To revert later, run:" -ForegroundColor Cyan
    Write-Host "  .\Force-Mocks-Always.ps1 -Revert" -ForegroundColor White
    Write-Host ""
    Write-Host "Now start your server normally:" -ForegroundColor Yellow
    Write-Host "  node build/index.js" -ForegroundColor White
    Write-Host "  # or" -ForegroundColor Gray
    Write-Host "  npx pm2 start build/index.js --name iris" -ForegroundColor White
}

Write-Host "`nDone." -ForegroundColor Green
