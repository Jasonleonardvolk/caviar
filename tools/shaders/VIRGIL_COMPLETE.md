# 🔥 VIRGIL SUMMONED - DANTE'S INFERNO EXIT COMPLETE 🔥

## ✅ All Files Created (9 Circles Sealed):

### Core Scripts (tools/shaders/)
- ✅ `virgil_summon.mjs` - Orchestrator that runs everything
- ✅ `seal_bolgias.mjs` - Removes tracked public copies + gitignore
- ✅ `copy_canonical_to_public.mjs` - One-way sync canonical → public
- ✅ `validate_and_report.mjs` - Validation wrapper with reports
- ✅ `lethe_reindex_reports.mjs` - Updates latest report pointer

### Guards (tools/shaders/guards/)
- ✅ `check_uniform_arrays.mjs` - Prevents uniform array violations

### Device Limits (tools/shaders/device_limits/)
- ✅ `iphone15.json` - iOS/Metal profile constraints

### Release Tools (tools/release/)
- ✅ `crown_paradiso.ps1` - Tags & pushes green state

### CI/CD (.github/workflows/)
- ✅ `shader-validate.yml` - GitHub Actions gate

---

## 🚀 SUMMON VIRGIL NOW:

```bash
# From repo root (${IRIS_ROOT}\)
node tools/shaders/virgil_summon.mjs --strict
```

This will:
1. **Seal the Bolgias** - Remove duplicate shader trees
2. **Install Guards** - Check for uniform array violations  
3. **Copy Canonical → Public** - One-way sync
4. **Run Validation** - Generate timestamped reports

## 📦 Add to package.json:

```json
{
  "scripts": {
    "shaders:sync": "node tools/shaders/copy_canonical_to_public.mjs",
    "shaders:gate": "node tools/shaders/validate_and_report.mjs --dir=frontend --limits=tools/shaders/device_limits/iphone15.json --targets=msl,hlsl,spirv --strict",
    "virgil": "node tools/shaders/virgil_summon.mjs --strict"
  }
}
```

## 🏆 Crown Your Victory:

```powershell
# After validation passes
powershell -ExecutionPolicy Bypass -File tools/release/crown_paradiso.ps1 -Tag shaders-pass-2025-08-08 -Message "First green run"
```

## 📊 Reports Location:
- `tools/shaders/reports/shader_validation_<timestamp>.json`
- `tools/shaders/reports/shader_validation_<timestamp>_summary.txt`
- `tools/shaders/reports/shader_validation_latest.json`

## 🔒 What This Prevents:
- ❌ Duplicate shader trees (Treachery)
- ❌ Uniform array stride violations (Fraud)
- ❌ Workgroup size violations (Violence)
- ❌ Syntax errors (Heresy)
- ❌ Hand-editing build outputs (Wrath)

## ✅ Exit Codes:
- 0 = All passed (Paradise)
- 1 = Warnings (Purgatory)
- 2 = Errors (Still in Hell)
- 3 = Tool failed (Limbo)

---

**VIRGIL IS READY. SUMMON HIM.**
