# HARDENINGS APPLIED - Complete Summary

## Date: 2025-08-16
## Based on: Review of Claude's fixes

## Three Critical Hardenings Applied

### 1. Cache-Bust Before Build ✅
**Problem:** `.svelte-kit` and Vite caches can hold old endpoint code, causing stale builds
**Solution:** Remove caches before building to force fresh compilation
```powershell
if (Test-Path .\.svelte-kit) { Remove-Item .\.svelte-kit -Recurse -Force }
if (Test-Path .\node_modules\.vite) { Remove-Item .\node_modules\.vite -Recurse -Force }
```
**Files Updated:**
- `Bulletproof-Build-And-Ship.ps1` - Includes cache-busting
- `Final-Runbook-Clean.ps1` - Implements full sequence

### 2. PM2 Environment Handling ✅
**Problem:** PM2 can inherit old environment variables from previous runs
**Solution:** Use `--update-env` flag and explicitly set environment before starting
```powershell
# Ensure runtime env is present
if (-not $env:PORT) { $env:PORT = "3000" }
if (-not $env:IRIS_USE_MOCKS) { $env:IRIS_USE_MOCKS = "1" }

# Start with --update-env
pm2 start .\build\index.js --name iris --update-env --time --restart-delay 3000
```
**Files Updated:**
- `tools\release\Reset-And-Ship.ps1` - Added --update-env flag

### 3. Clean Self-Test Block ✅
**Problem:** Parser errors due to unbalanced quotes/braces in test block
**Solution:** Replaced with clean, balanced test function
```powershell
function Test-Url {
  param([string]$Url, [int]$ExpectedStatus = 200)
  try {
    $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -Method GET -TimeoutSec 10
    return ($resp.StatusCode -eq $ExpectedStatus)
  } catch { return $false }
}
```
**Files Updated:**
- `tools\release\Reset-And-Ship.ps1` - Clean test block

### 4. Renderer SSR Guard (Already Applied) ✅
**File:** `src\routes\renderer\+page.server.ts`
**Purpose:** Prevents 500 errors during SSR by returning mock data when mocks enabled

## Scripts Created

### Primary Scripts
1. **Bulletproof-Build-And-Ship.ps1**
   - Complete implementation with all hardenings
   - Cache-bust → Build → Launch → Test
   - Supports both mock and real modes

2. **Final-Runbook-Clean.ps1**
   - Exact implementation of review's runbook
   - Clean & deterministic sequence
   - Ready for production use

3. **Verify-Hardenings.ps1**
   - Checks all hardenings are in place
   - Validates source files, compiled output, environment
   - Provides clear pass/fail status

### Supporting Scripts (from earlier)
- `Apply-All-Fixes.ps1` - Applies all fixes comprehensively
- `Quick-Test-Mocks.ps1` - Quick mock endpoint testing

## Verification Checklist

Run `.\Verify-Hardenings.ps1` to check:
- ✅ Mock gates in API endpoints
- ✅ Renderer SSR guard exists
- ✅ PM2 --update-env in Reset-And-Ship.ps1
- ✅ Clean self-test block
- ✅ Compiled output contains mocks (after build)
- ✅ Environment configuration

## Usage Instructions

### Quick Verification
```powershell
cd D:\Dev\kha\tori_ui_svelte
.\Verify-Hardenings.ps1
```

### Bulletproof Deployment
```powershell
cd D:\Dev\kha\tori_ui_svelte
.\Bulletproof-Build-And-Ship.ps1 -Mode mock -UsePM2
```

### Clean Runbook (Recommended)
```powershell
cd D:\Dev\kha\tori_ui_svelte
.\Final-Runbook-Clean.ps1
```

### Manual Steps
```powershell
# 0) Cache-bust
if (Test-Path .\.svelte-kit) { Remove-Item .\.svelte-kit -Recurse -Force }
if (Test-Path .\node_modules\.vite) { Remove-Item .\node_modules\.vite -Recurse -Force }

# 1) Build
pnpm install
pnpm run build

# 2) Verify mocks compiled
Select-String -Path `
  ".\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js", `
  ".\.svelte-kit\output\server\entries\endpoints\api\memory\state\_server.ts.js" `
  -Pattern "note:\"mock\"|note:'mock'"

# 3) Ship with PM2
$env:IRIS_USE_MOCKS = "1"
$env:PORT = "3000"
.\tools\release\Reset-And-Ship.ps1 -UsePM2

# 4) Test
Invoke-RestMethod http://127.0.0.1:3000/api/pdf/stats
Invoke-RestMethod http://127.0.0.1:3000/api/memory/state
```

## Expected Results

With all hardenings applied:
- Build always uses fresh code (no cache issues)
- PM2 always gets correct environment variables
- Tests run without parser errors
- Endpoints return mock data with `note: "mock"`
- No 500 errors on `/` or `/renderer`

## Git Commit

```bash
git add `
  .\tools\release\Reset-And-Ship.ps1 `
  .\src\routes\api\pdf\stats\+server.ts `
  .\src\routes\api\memory\state\+server.ts `
  .\src\routes\renderer\+page.server.ts `
  .\Bulletproof-Build-And-Ship.ps1 `
  .\Final-Runbook-Clean.ps1 `
  .\Verify-Hardenings.ps1

git commit -m 'ship: cache-bust + pm2 --update-env; api: compile-safe mocks; renderer: SSR mock guard'
```

## Troubleshooting

If endpoints still don't return mock data after all fixes:
1. Check first 20 lines of compiled files:
   ```powershell
   Get-Content ".\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js" -Head 20
   ```
2. Check PM2 logs:
   ```powershell
   pm2 logs iris --lines 80
   ```
3. Verify environment in PM2:
   ```powershell
   pm2 env iris
   ```

## Status: READY TO SHIP ✅

All hardenings are in place. The system is bulletproof for mock deployment.
