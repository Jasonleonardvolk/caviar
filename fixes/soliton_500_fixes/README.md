# Soliton API 500 Error Fixes

This directory contains all the scripts needed to fix the 500 errors on `/api/soliton/*` endpoints.

## Quick Start

### 1. Test if you have the problem
```bash
python test_soliton_api.py
```

If you see 500 errors or connection failures, continue to step 2.

### 2. Apply all fixes
```bash
python fix_soliton_main.py
```

This will:
- Set environment variables (TORI_DISABLE_MESH_CHECK=1)
- Fix Pydantic v2 imports throughout the codebase
- Fix asyncio.run() issues in the PDF pipeline
- Consolidate duplicate pipeline files
- Install required dependencies

### 3. Apply the fixed API route
```bash
python apply_soliton_api_fix.py
```

This replaces the broken soliton.py with the fixed version.

### 4. Start the backend with debug logging
```bash
python start_backend_debug.py
```

### 5. Test again (in another terminal)
```bash
python test_soliton_api.py
```

You should now see all tests passing!

## Files in this directory

- `fix_soliton_main.py` - Main fix script that applies all fixes
- `soliton_fixed.py` - The fixed API route file
- `apply_soliton_api_fix.py` - Script to apply the fixed API route
- `test_soliton_api.py` - Test script to verify endpoints work
- `start_backend_debug.py` - Start backend with debug logging
- `README.md` - This file

## What gets fixed

1. **Concept-mesh import errors** - Falls back to stub implementation when not available
2. **Pydantic v2 compatibility** - Fixes BaseSettings imports
3. **asyncio.run() in event loops** - Properly handles async contexts
4. **Duplicate pipelines** - Consolidates to single canonical location
5. **API error handling** - Returns proper status codes instead of 500s

## Manual steps still needed

1. Apply frontend guards to prevent cascading failures in `solitonMemory.ts`
2. Optionally build concept_mesh with maturin for full functionality

## Troubleshooting

If you still see errors after applying fixes:

1. Check the diagnostic endpoint: http://localhost:5173/api/soliton/diagnostic
2. Verify environment variable: `echo %TORI_DISABLE_MESH_CHECK%`
3. Check Python imports: `python -c "from mcp_metacognitive.core import soliton_memory"`
4. Review backend logs for detailed error messages

## Success indicators

After all fixes are applied:
- `/api/soliton/health` returns 200 OK
- `/api/soliton/init` returns success (mock or real)
- `/api/soliton/stats/{user_id}` returns valid stats
- No 500 errors in the logs
- Frontend "Memory initializing..." banner disappears
