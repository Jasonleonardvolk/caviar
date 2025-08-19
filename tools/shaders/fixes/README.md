# 🔧 Shader Auto-Fixes

Automated fixes for common WGSL shader validation warnings.

## 📁 Files

- `fix_vec3_alignment.mjs` - Converts vec3→vec4 in storage buffers for alignment
- `fix_bounds_checking.mjs` - Adds array bounds checking with clamp_index()
- `fix_const_let.mjs` - Changes immutable let to const
- `fix_all.mjs` - Runs all fixes in the correct order
- `add_fix_scripts.mjs` - Adds npm scripts to package.json

## 🚀 Quick Start

### First Time Setup
```bash
# Add scripts to package.json
node tools/shaders/fixes/add_fix_scripts.mjs
```

### Fix All Issues
```bash
# Run all fixes, sync, and validate
npm run shaders:fix
```

## 📋 Individual Fixes

### Vec3 Storage Alignment
**Problem:** vec3 in storage buffers causes alignment issues (needs 16-byte alignment)
```bash
npm run shaders:fix:vec3
```
**Solution:** Converts `vec3<f32>` to `vec4<f32>` (uses .xyz, .w for padding)

### Array Bounds Checking  
**Problem:** Dynamic array access without bounds checking
```bash
npm run shaders:fix:bounds
```
**Solution:** Adds `clamp_index()` helper and wraps all dynamic array accesses

### Const vs Let
**Problem:** Using `let` for immutable values
```bash
npm run shaders:fix:const
```
**Solution:** Changes immutable `let` declarations to `const`

## 🔄 Workflow

1. **Run fixes**
   ```bash
   npm run shaders:fix:all
   ```

2. **Sync to public**
   ```bash
   npm run shaders:sync
   ```

3. **Validate**
   ```bash
   npm run shaders:gate:iphone
   ```

Or all at once:
```bash
npm run shaders:fix
```

## 📁 Backups

Each fix creates backup files:
- `*.pre-vec4.bak` - Before vec3→vec4 conversion
- `*.pre-bounds.bak` - Before bounds checking
- `*.pre-const.bak` - Before const/let fixes

### Restore from Backups
```powershell
# PowerShell - restore all backups
Get-ChildItem -Path "frontend\lib\webgpu\shaders" -Filter "*.bak" | 
  ForEach { Copy-Item $_.FullName ($_.FullName -replace "\.bak$", "") }
```

## 🎯 What Gets Fixed

### Storage Buffer Alignment (2 warnings)
- `avatarShader.wgsl` - vec3 fields in Particle struct

### Dynamic Array Bounds (146 warnings)
- All files with array indexing using variables
- Adds safety for edge cases

### Const Optimization (3 warnings)
- `lenticularInterlace.wgsl` - subpixel_width
- `propagation.wgsl` - view_angle  
- `velocityField.wgsl` - momentum, value

## ✅ Expected Results

After running fixes:
- **0 Failed** (no more syntax/semantic errors)
- **0 Warnings** (all issues resolved)
- **17 Passed** (all shaders valid)

## 🔍 Validation Profiles

Test against different device limits:
```bash
npm run shaders:gate:iphone   # Mobile (256 threads)
npm run shaders:gate:desktop  # Desktop (1024 threads)
npm run shaders:gate:low      # Low-end (256 threads)
```

## 🛠️ Maintenance

If new shaders are added:
1. Run `npm run shaders:fix:all` to apply fixes
2. Commit both shader and backup files
3. CI will validate on push

## 📊 Before & After

**Before:**
- 6 Failed (syntax errors)
- 11 Files with warnings
- 148 Total warnings
- 0 Passed

**After fixes:**
- 0 Failed ✅
- 0 Warnings ✅
- 17 Passed ✅
