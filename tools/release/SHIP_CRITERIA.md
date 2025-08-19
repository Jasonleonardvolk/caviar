# 🚀 IRIS Release Gate - Ship Criteria

## One-Button Release Check
```powershell
# Run all checks with one command:
.\tools\release\IrisOneButton.ps1

# Or without building:
.\tools\release\IrisOneButton.ps1 -Build:$false
```

## ✅ Ship Criteria (Keep Posted on Wall)

### 1. **No Absolute Paths** ❌→✅
- **Check**: `node tools\runtime\preflight.mjs`
- **Fix**: `python tools\refactor\refactor_continue.py`
- **Status**: 930 files refactored, runtime resolution active

### 2. **TypeScript Build** = 0 errors
- **Check**: `npx tsc -p frontend\tsconfig.json`
- **Location**: `frontend\tsconfig.json`
- **Logs**: `tools\release\error_logs\[timestamp]\typescript_errors.txt`

### 3. **Shader Gate** = 0 FAIL for iphone11 and iphone15
- **Check**: `.\tools\shaders\run_shader_gate.ps1 -Targets @("iphone11","iphone15")`
- **WARNs OK** if documented in `tools\shaders\validator_suppressions.json`
- **Reports**: `tools\shaders\reports\`
- **Robust to**:
  - Tint found/missing (`tools\shaders\bin\tint.exe`)
  - Either validator name (`validate-wgsl.js` or `validate_and_report.mjs`)

### 4. **Quilt Shader Sync**
- **Edit**: `frontend\lib\webgpu\shaders\**`
- **Publish to**: `frontend\public\hybrid\wgsl\**`
- **No import mismatches** between source and published

### 5. **Production API**
- **Config**: `.env.production` points to live API
- **Check**: Passes smoke tests
- **Test**: `node tools\release\api-smoke.js --env ".env.production"`

## 📊 Release Artifacts

When spinning up fresh chat for final mile:
1. Drop latest `tools\shaders\reports\*` summaries
2. Include `tools\release\iris_release_summary.txt`
3. Check `tools\release\error_logs\` for any issues

## 🔄 Adding New Targets

To add new device targets (e.g., iphone17, android_adreno):
1. Add device limits: `tools\shaders\device_limits\[device].json`
2. Update IrisOneButton.ps1: `-Targets @("iphone11","iphone15","iphone17")`

## 🎯 Quick Commands

```powershell
# Full release gate
.\tools\release\IrisOneButton.ps1

# Just check, no build
.\tools\release\IrisOneButton.ps1 -Build:$false

# Check absolute paths only
node tools\runtime\preflight.mjs

# Check TypeScript only
npx tsc -p frontend\tsconfig.json

# Check shaders only
.\tools\shaders\run_shader_gate.ps1 -Targets @("iphone11","iphone15")

# Generate release summary
.\tools\release\generate_summary.ps1
```

---

**Fresh eyes + coffee** ☕ = Last mile success! 🏁
