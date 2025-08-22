# MOCK ENDPOINTS FIX - Complete Solution

## The Problem
When running the server, the API endpoints return:
```json
{"ok":false,"error":"PDF processing service not configured","hint":"Set IRIS_USE_MOCKS=1 to use mock data"}
```

Even though `IRIS_USE_MOCKS=1` is set, the mocks aren't working.

## Root Cause
The issue occurs because:
1. **SvelteKit/Vite caching** - Old compiled output without mock gates is cached
2. **Environment variables not propagating** - Build or runtime isn't seeing `IRIS_USE_MOCKS=1`
3. **Build artifacts stale** - The `.svelte-kit` folder contains old compiled code

## Solutions (Try in Order)

### Solution 1: Quick Start with Mocks
The simplest approach - ensures environment is set correctly:
```powershell
cd D:\Dev\kha\tori_ui_svelte
.\Start-With-Mocks.ps1
```
This script:
- Sets `.env.local` with `IRIS_USE_MOCKS=1`
- Starts server with environment variables explicitly set
- Tests endpoints automatically

### Solution 2: Force Fresh Build
If Solution 1 doesn't work, force a completely fresh build:
```powershell
cd D:\Dev\kha\tori_ui_svelte
.\Force-Fresh-Build.ps1
```
This script:
- Aggressively clears ALL caches
- Rebuilds from scratch
- Verifies mocks are in compiled output
- Starts server with correct environment

### Solution 3: Debug the Issue
To understand what's wrong:
```powershell
cd D:\Dev\kha\tori_ui_svelte
.\Debug-Mock-Issue.ps1
```
This will show:
- Current environment variables
- Contents of .env files
- What's in the compiled output
- Whether server is seeing the mocks

### Solution 4: Manual Fix
If scripts don't work, manually fix:

```powershell
# 1. Stop everything
Stop-Job -Name "iris-*" -ErrorAction SilentlyContinue
npx pm2 stop all

# 2. Set environment
$env:IRIS_USE_MOCKS = "1"
$env:PORT = "3000"
$env:IRIS_ALLOW_UNAUTH = "1"

# 3. Create .env.local
@"
PORT=3000
IRIS_USE_MOCKS=1
IRIS_ALLOW_UNAUTH=1
NODE_ENV=production
"@ | Out-File -FilePath ".env.local" -Encoding UTF8

# 4. Clear ALL caches
Remove-Item .\.svelte-kit -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\build -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item .\node_modules\.vite -Recurse -Force -ErrorAction SilentlyContinue

# 5. Rebuild
pnpm install
pnpm run build

# 6. Start with environment inline
$env:IRIS_USE_MOCKS = "1"; node build/index.js
```

## Verification

After starting, test the endpoints:
```powershell
# Should return mock data with note: "mock"
Invoke-RestMethod http://127.0.0.1:3000/api/pdf/stats
Invoke-RestMethod http://127.0.0.1:3000/api/memory/state
```

Expected response:
```json
{
  "ok": true,
  "docs": 3,
  "pages": 42,
  "note": "mock"
}
```

## Key Files to Check

1. **Source files** (should have mock gates):
   - `src\routes\api\pdf\stats\+server.ts`
   - `src\routes\api\memory\state\+server.ts`
   
   Should contain:
   ```typescript
   if (env.IRIS_USE_MOCKS === '1') {
     return json({ ok: true, docs: 3, pages: 42, note: 'mock' });
   }
   ```

2. **Environment file** (`.env.local`):
   ```
   IRIS_USE_MOCKS=1
   ```

3. **Compiled output** (after build):
   - `.\.svelte-kit\output\server\entries\endpoints\api\pdf\stats\_server.ts.js`
   
   Should contain references to `IRIS_USE_MOCKS` and mock responses.

## Common Issues

### Issue: "Cannot overwrite variable PID"
**Fix:** Already fixed - scripts now use `$procId` instead of `$pid`

### Issue: "pm2 not recognized"
**Fix:** Already fixed - scripts now use `npx pm2` and install locally if needed

### Issue: Mocks still not working after rebuild
**Possible causes:**
1. Node.js process caching modules - restart completely
2. Port 3000 blocked - check with `Get-NetTCPConnection -LocalPort 3000`
3. Environment not set in running process - use inline env vars

## The Nuclear Option

If nothing else works:
```powershell
# 1. Complete cleanup
Get-Job | Stop-Job
Get-Job | Remove-Job
npx pm2 kill
Remove-Item .\.svelte-kit -Recurse -Force
Remove-Item .\build -Recurse -Force
Remove-Item .\node_modules -Recurse -Force

# 2. Fresh install
pnpm install

# 3. Build with mocks forced
$env:IRIS_USE_MOCKS = "1"
pnpm run build

# 4. Run with mocks forced
$env:IRIS_USE_MOCKS = "1"
$env:PORT = "3000"
node build/index.js
```

## Status

With these scripts, the mock endpoints should work. The key is:
1. **Clear caches** before building
2. **Set environment** during both build and runtime
3. **Use .env.local** for persistent settings
4. **Verify compiled output** contains mock checks

Run `.\Start-With-Mocks.ps1` for the quickest solution!
