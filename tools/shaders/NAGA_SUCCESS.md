# 🎉 NAGA INSTALLED & CONFIGURED!

## ✅ What's Ready:

1. **Naga WGSL Validator** - Installed at `tools\shaders\bin\naga.exe`
2. **Tint Compatibility Layer** - Scripts expecting `tint.exe` will use Naga
3. **Validation Pipeline** - Ready to run

## 🧪 Test Naga:

```cmd
cd ${IRIS_ROOT}\tools\shaders\bin
.\test_naga.cmd
```

## 🚀 RUN VIRGIL NOW:

```bash
# Go to repo root
cd ${IRIS_ROOT}

# Add scripts to package.json (if not done yet)
node tools/shaders/add_virgil_scripts.mjs

# SUMMON VIRGIL - This will validate all shaders!
npm run virgil
```

Or directly:
```bash
cd ${IRIS_ROOT}
node tools/shaders/virgil_summon.mjs --strict
```

## 📊 What Virgil Will Do:

1. **Seal Bolgias** - Remove duplicate shader trees
2. **Check Guards** - Verify no uniform array violations  
3. **Sync Shaders** - Copy canonical → public
4. **Validate All** - Run Naga on every WGSL file
5. **Generate Reports** - Create timestamped validation reports

## 📁 Reports Location:
- `tools\shaders\reports\shader_validation_<timestamp>.json`
- `tools\shaders\reports\shader_validation_<timestamp>_summary.txt`
- `tools\shaders\reports\shader_validation_latest.json`

## ✅ Compatibility Notes:

- Naga validates WGSL syntax and semantics
- The `tint.bat` wrapper makes Naga work with scripts expecting Tint
- Naga can convert to SPIR-V: `naga.exe shader.wgsl output.spv`
- For HLSL/MSL conversion, Naga validates but doesn't output (that's OK for validation)

## 🏆 Victory Conditions:

After running Virgil, if you see:
- **Exit code 0** = All shaders pass! (Paradise) 🎉
- **Exit code 1** = Warnings only (Purgatory) ⚠️
- **Exit code 2** = Errors found (Still climbing) ❌

## 🔥 READY TO ESCAPE HELL:

```bash
npm run virgil
```

**GO FOR IT!**
