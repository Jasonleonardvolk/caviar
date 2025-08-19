# TORI/KHA Launch Issues - Quick Fix Guide

## 🚨 Issues Identified

1. **Pydantic v2 Migration Error**
   - `BaseSettings` has been moved to `pydantic-settings` package
   - This breaks imports in multiple files

2. **Missing SolitonMemoryLattice**
   - The import `from core.soliton_memory import SolitonMemoryLattice` fails
   - Class doesn't exist in soliton_memory.py

3. **Frontend Proxy Errors**
   - Vite proxy can't find `/api/soliton/*` endpoints
   - API returns 404 for these routes

4. **Optional Components Missing**
   - Penrose, TONKA, concept_mesh libraries not compiled/installed
   - These are optional but cause warning messages

## 🔧 Quick Fix Steps

### Step 1: Run the Comprehensive Fix Script
```bash
python fix_all_issues.py
```

This script will:
- Update all requirements files to include `pydantic-settings`
- Install the required packages
- Fix Pydantic imports across all Python files
- Create missing Soliton API endpoints
- Set up proper initialization files

### Step 2: Install Dependencies Manually (if needed)
```bash
pip install pydantic pydantic-settings --upgrade
```

### Step 3: Restart the Launcher
```bash
python enhanced_launcher.py
```

## 🛠️ Manual Fixes (if automatic fix doesn't work)

### Fix Pydantic Imports
Run the dedicated Pydantic fix script:
```bash
python fix_pydantic_imports.py
```

### Fix SolitonMemoryLattice Import
The fix script already created `mcp_metacognitive/core/__init__.py` with the proper exports.

### Fix Soliton API Endpoints
The fix script created `api/routes/soliton.py` with stub endpoints. If your API file doesn't include it automatically, add:

```python
from api.routes.soliton import router as soliton_router
app.include_router(soliton_router)
```

## 📋 Verification

After fixes, you should see:
- ✅ No more Pydantic import errors
- ✅ MCP Metacognitive loads (even if limited)
- ✅ Frontend proxy errors stop (returns actual data)
- ✅ System starts with only optional component warnings

## 🎯 Expected Output After Fixes

The launcher should show:
```
✅ System resources are healthy
✅ Found available API port: 8002
✅ Frontend started successfully!
🎯 ENHANCED TORI SYSTEM READY
```

With only these acceptable warnings:
- ⚠️ Penrose not available (optional)
- ⚠️ TONKA not available (optional)
- ⚠️ Some concept mesh features limited (but working)

## 💡 Tips

1. The Penrose/TONKA warnings are OPTIONAL - system works without them
2. Frontend proxy errors will stop once Soliton endpoints exist
3. Always check `api_port.json` for the current API port
4. Logs are in `logs/session_[timestamp]/`

## 🚀 Next Steps

1. Run `python fix_all_issues.py`
2. Start with `python enhanced_launcher.py`
3. Open browser to http://localhost:5173
4. Upload a PDF to test the system!

Good luck! 🎉
