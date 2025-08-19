# 🚀 TORI No-DB Migration - Quick Start Guide

## One-Command Setup (Windows PowerShell)

```powershell
# Run from ${IRIS_ROOT}
python master_nodb_fix.py
.\setup_nodb_complete.ps1
```

## Manual Steps

### 1. Apply All Fixes
```bash
cd ${IRIS_ROOT}
python master_nodb_fix.py
```

This single script fixes EVERYTHING:
- ✅ Deprecated pd.io.json.dumps → json.dumps
- ✅ Missing datetime imports
- ✅ Import path standardization
- ✅ AST-based _last_betti initialization
- ✅ Bounded deque collections
- ✅ Optional WebSockets with cleanup
- ✅ Token rate limiting (MAX_TOKENS_PER_MIN)
- ✅ scipy import guards
- ✅ Enhanced migration scripts

### 2. Set Environment (PowerShell)
```powershell
$env:TORI_STATE_ROOT = "C:\tori_state"
$env:MAX_TOKENS_PER_MIN = "200"
$env:PYTHONPATH = "$PWD;$PWD\kha"
```

### 3. Validate
```bash
python alan_backend\validate_nodb_final.py
```

### 4. Run Tests
```bash
pytest alan_backend\test_nodb_migration.py
```

### 5. Start System
```bash
python alan_backend\start_true_metacognition.bat
```

## 📦 Files Modified

**Core Files:**
- `python/core/__init__.py`
- `python/core/torus_registry.py` 
- `python/core/torus_cells.py`
- `python/core/observer_synthesis.py`

**Runtime Modules:**
- `alan_backend/origin_sentry_modified.py`
- `alan_backend/eigensentry_guard_modified.py`
- `alan_backend/chaos_channel_controller_modified.py`
- `alan_backend/braid_aggregator_modified.py`

**Scripts:**
- `alan_backend/migrate_to_nodb_ast.py`
- `alan_backend/test_nodb_migration.py`
- `alan_backend/validate_nodb_final.py`

## 🎯 Success Indicators

Look for these in the logs:
- `✅ No-DB persistence components fully loaded`
- `✅ All validation checks passed!`
- No flake8-forbid-import violations
- Parquet files created in TORI_STATE_ROOT

## 🔥 That's it! The No-DB migration is complete and production-ready! 🔥
