# 🚀 TORI No-DB Migration v2 - Quick Start Guide

## What's New in v2

- **AST-based import fixes** - No more regex brittleness
- **Configurable canonical root** - Choose between `python.core` or `kha.python.core`
- **Complete RateLimiter implementation** - Proper token throttling
- **Cross-platform validation** - No Unix dependencies
- **Better error handling** - Clear feedback on issues
- **Requirements file** - All dependencies documented

## One-Command Setup (Windows PowerShell)

```powershell
# Run from ${IRIS_ROOT}
python master_nodb_fix_v2.py
.\setup_nodb_complete_v2.ps1
```

## Choose Your Import Root

The script now supports choosing your canonical import root:

```bash
# Use default (kha.python.core)
python master_nodb_fix_v2.py

# Or specify explicitly
python master_nodb_fix_v2.py --canonical-root python.core
```

## Manual Steps

### 1. Install Dependencies
```bash
pip install -r requirements_nodb.txt
```

### 2. Apply All Fixes
```bash
cd ${IRIS_ROOT}
python master_nodb_fix_v2.py
```

### 3. Set Environment (PowerShell)
```powershell
$env:TORI_STATE_ROOT = "C:\tori_state"
$env:MAX_TOKENS_PER_MIN = "200"
$env:PYTHONPATH = "$PWD;$PWD\kha"
```

### 4. Validate
```bash
python alan_backend\validate_nodb_final.py
```

### 5. Run Tests
```bash
pytest alan_backend\test_nodb_migration.py
```

## 🔧 What Gets Fixed

**AST-Based Fixes:**
- Import path standardization (handles multi-line imports)
- datetime import additions (proper placement)
- WebSocket optional imports with cleanup methods
- _last_betti initialization (no docstring issues)

**Robust Implementations:**
- Complete RateLimiter class for token throttling
- Cross-platform validation (no grep dependency)
- Improved PowerShell script with error handling
- Scipy import guards with fallback

**Configuration:**
- Canonical import root (configurable)
- Environment-based thresholds
- Rate limiting controls

## 📦 Files Modified

All files from v1 plus:
- `requirements_nodb.txt` - All dependencies listed
- `validate_nodb_final.py` - Cross-platform compatible
- `setup_nodb_complete_v2.ps1` - Better file discovery

## 🎯 Success Indicators

- ✅ All AST transformations successful
- ✅ No database imports detected
- ✅ No pd.io.json usage found
- ✅ WebSockets are optional
- ✅ Collections are bounded
- ✅ Rate limiting active

## 🆕 Key Improvements from v1

1. **AST over Regex** - More reliable code transformations
2. **Platform Independence** - Works on Windows/Linux/Mac
3. **Complete RateLimiter** - No missing implementation
4. **Better Error Messages** - Clear feedback on failures
5. **Dependency Management** - requirements.txt included

## 🔥 Ready for production deployment! 🔥
