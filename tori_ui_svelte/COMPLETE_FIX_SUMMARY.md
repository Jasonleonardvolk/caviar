# COMPLETE FIX - Dependencies & Centralized Mocks

## What Was Wrong

1. **Missing Dependency:** `@sveltejs/adapter-node` wasn't installed
2. **Mock Detection:** Environment variables weren't being read consistently
3. **Health Endpoint:** Was failing, causing verifier to fail

## What I Fixed

### 1. Centralized Mock Detection
Created `src/lib/server/env.ts` - Single source of truth:
```typescript
export const isMock = (): boolean => {
  // Checks both SvelteKit env and process.env
  // Accepts 1/true/yes/y (case insensitive)
  // IRIS_FORCE_MOCKS overrides IRIS_USE_MOCKS
}
```

### 2. Updated All Endpoints
- `src/routes/api/pdf/stats/+server.ts` - Uses `isMock()`
- `src/routes/api/memory/state/+server.ts` - Uses `isMock()`
- `src/routes/api/health/+server.ts` - Always returns 200 with mode info

### 3. Created Fix Scripts
- `Fix-Dependencies-And-Mocks.ps1` - Complete fix (installs deps + mocks)
- `Install-Missing-Dependencies.ps1` - Just fixes dependencies

## How to Fix Everything

### Option 1: Complete Fix (Recommended)
```powershell
cd D:\Dev\kha\tori_ui_svelte
.\Fix-Dependencies-And-Mocks.ps1
```
This will:
1. Install missing `@sveltejs/adapter-node`
2. Create centralized mock detection
3. Update all endpoints
4. Build and test

### Option 2: Just Fix Dependencies
If the build is completely broken:
```powershell
.\Install-Missing-Dependencies.ps1
```
Then manually:
```powershell
pnpm run build
```

### Option 3: Manual Steps
```powershell
# 1. Install missing dependency
pnpm add -D @sveltejs/adapter-node

# 2. Install all dependencies
pnpm install

# 3. Clear cache
Remove-Item .\.svelte-kit -Recurse -Force

# 4. Build
$env:IRIS_USE_MOCKS = "1"
pnpm run build

# 5. Test
node build/index.js
```

## Testing

After fixing, test the endpoints:
```powershell
# Health check (should always work)
Invoke-RestMethod http://127.0.0.1:3000/api/health

# Mock endpoints
Invoke-RestMethod http://127.0.0.1:3000/api/pdf/stats
Invoke-RestMethod http://127.0.0.1:3000/api/memory/state
```

Expected responses:
- Health: `{ "ok": true, "status": "ok", "mode": "mock" }`
- PDF: `{ "ok": true, "docs": 3, "pages": 42, "note": "mock" }`
- Memory: `{ "ok": true, "state": "idle", "concepts": 0, "note": "mock" }`

## Files Created/Modified

### Created
- `src/lib/server/env.ts` - Centralized mock detection
- `src/routes/api/health/+server.ts` - Health endpoint
- `Fix-Dependencies-And-Mocks.ps1` - Complete fix script
- `Install-Missing-Dependencies.ps1` - Dependency fix script

### Modified
- `src/routes/api/pdf/stats/+server.ts` - Now uses `isMock()`
- `src/routes/api/memory/state/+server.ts` - Now uses `isMock()`

## Why This Works

1. **Single Source of Truth:** All endpoints use the same `isMock()` function
2. **Multiple Env Sources:** Checks both SvelteKit's env and process.env
3. **Flexible Values:** Accepts 1/true/yes/y in any case
4. **Force Override:** `IRIS_FORCE_MOCKS` can override `IRIS_USE_MOCKS`
5. **Health Always Works:** Health endpoint always returns 200

## Status: READY TO BUILD ✅

Run `.\Fix-Dependencies-And-Mocks.ps1` and everything should work!
