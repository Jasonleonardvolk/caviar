# TORI Gremlin Hunt - Action Plan

## 🚨 PRIMARY BLOCKER (Build-breaking)

### Issue: Duplicate export of `updateSystemEntropy`
**Status**: PARTIALLY FIXED - Need verification

**To verify the fix:**
```bash
cd ${IRIS_ROOT}
python check_exports.py
```

This will show if there are still duplicate exports. If it shows only 1 export, the fix worked!

**If still broken**, manually check around line 1048 in `conceptMesh.ts`

## 🧹 Clear Vite Cache (Just in case)

```bash
cd ${IRIS_ROOT}\tori_ui_svelte
rmdir /s /q .svelte-kit
rmdir /s /q node_modules\.vite
npm run dev
```

## 📑 SECONDARY ISSUES (Non-blocking)

### 1. UTF-8 Decode Error
**File**: `file_storage`
**Fix**:
```bash
cd ${IRIS_ROOT}
python fix_encoding_issues.py
```

### 2. Missing Python Modules
**Fix PYTHONPATH**:
```powershell
$env:PYTHONPATH = "${IRIS_ROOT};$env:PYTHONPATH"
python -m tori.enhanced_launcher
```

## 🎯 Quick Test Sequence

1. **Check for duplicate exports**:
   ```bash
   python check_exports.py
   ```

2. **If only 1 export found, restart**:
   ```bash
   python enhanced_launcher.py
   ```

3. **Check browser**:
   - Go to http://localhost:5173
   - Open console (F12)
   - Should NOT see "Multiple exports" error

4. **If still seeing errors**, check Vite output in terminal

## 🔍 What Success Looks Like

✅ Vite shows clean startup:
```
VITE v5.4.19 ready in XXX ms
➜ Local: http://localhost:5173/
```

✅ No red error overlays in browser

✅ Chat shows "Memory: Ready" instead of "Memory: Initializing"

## 💡 Remember

The Python warnings (concept mesh, MCP, etc.) are **secondary** - they won't stop the UI from working. Focus on getting Vite to compile first!
