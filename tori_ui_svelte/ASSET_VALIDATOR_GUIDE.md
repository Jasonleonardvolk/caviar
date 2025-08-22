# 🔍 Asset Manifest Validator v2.0

## Overview

The enhanced asset validator ensures all 3D models in your AR assets system meet quality and performance standards. It inspects actual GLTF/GLB files to verify triangle counts, file integrity, and manifest accuracy.

## 🚦 What It Validates

### Required Checks (Blocks on Failure)
- ✅ **File Existence**: Verifies all `opt` paths point to real files
- ✅ **File Format**: Ensures files are `.glb` or `.gltf` format
- ✅ **Triangle Count**: Counts actual triangles in each model
- ✅ **Triangle Limit**: Enforces max 100,000 triangles per asset (configurable)
- ✅ **Required Fields**: Validates manifest has all required keys
- ✅ **Non-Zero Geometry**: Ensures models have valid triangle data

### Optional Warnings
- ⚠️ **Brandless Flag**: Warns if `brandless` field is missing
- ⚠️ **Scale Information**: Warns if `scale_m` is not specified
- ⚠️ **Polycount Accuracy**: Warns if `polycount_hint` differs >20% from actual

## 🎯 Triangle Limits

| Limit | Use Case | Performance Target |
|-------|----------|-------------------|
| 10,000 | Mobile AR (Low-end) | 60 FPS on most devices |
| 50,000 | Mobile AR (Standard) | 30-60 FPS on modern phones |
| **100,000** | **Default Limit** | **Good performance on iPhone 11+** |
| 200,000 | Desktop/High-end | May struggle on mobile |

## 📋 Usage

### NPM Script (Recommended)
```bash
# Run with default 100k triangle limit
pnpm run validate:assets

# Quick test
pnpm test
```

### Direct Command
```bash
# Default (100k triangles max)
node tools/assets/validate-manifest.mjs assets/3d/luxury/ASSET_MANIFEST.json

# Custom limit (50k triangles)
node tools/assets/validate-manifest.mjs assets/3d/luxury/ASSET_MANIFEST.json --maxTris=50000

# High limit (200k triangles)
node tools/assets/validate-manifest.mjs assets/3d/luxury/ASSET_MANIFEST.json --maxTris=200000
```

### Batch File
```cmd
# Run validator test
.\test-asset-validator.bat
```

## 🔒 Git Integration

The validator is **automatically run on every `git push`** through the pre-push hook:

1. **Pre-push hook** calls → `verify-setup.ps1`
2. **verify-setup.ps1** runs → `validate-manifest.mjs`
3. **If validation fails** → Push is blocked
4. **If validation passes** → Push continues

### Bypass (Emergency Only!)
```bash
# Skip all pre-push checks
git push --no-verify
```

## 📊 Output Example

### Success Output
```
=====================================
    Asset Manifest Validator v2.0
=====================================
Manifest: D:\Dev\kha\tori_ui_svelte\assets\3d\luxury\ASSET_MANIFEST.json
Max triangles per asset: 100,000

Checking cgtrader_generic_sneaker...
  [OK] 15,234 triangles
Checking skfb_lowpoly_watch...
  [OK] 1,728 triangles
Checking skfb_perfume_bottle...
  [OK] 24,892 triangles

=====================================
Validation PASSED:
  Assets validated: 3
  Total triangles: 41,854
  Average per asset: 13,951
  Highest poly count: 24,892 (skfb_perfume_bottle)
  Triangle limit: 100,000
=====================================
[OK] All assets within limits!
```

### Failure Output
```
=====================================
Validation FAILED:
  [X] item[0] id=heavy_model: triangle count 150,000 exceeds limit 100,000
  [X] item[1] id=broken_model: opt file not found: assets/3d/luxury/broken/opt/model.glb
=====================================
```

## 🛠️ Troubleshooting

### "Manifest not found"
- Ensure you're in the project root: `D:\Dev\kha\tori_ui_svelte`
- Check file exists: `assets\3d\luxury\ASSET_MANIFEST.json`

### "Model files not yet downloaded"
- This is a warning, not an error
- Download the free models from the links in AR_ASSETS_GUIDE.md
- Or create placeholder files for testing

### "Triangle count exceeds limit"
- Model is too complex for mobile AR
- Options:
  1. Decimate the model in Blender/3D software
  2. Use a different model
  3. Increase limit (not recommended for mobile)

### "Node.js not found"
- Install Node.js from https://nodejs.org
- Ensure it's in your PATH

## 🎮 Testing Different Scenarios

### Test with Strict Mobile Limit
```bash
node tools/assets/validate-manifest.mjs assets/3d/luxury/ASSET_MANIFEST.json --maxTris=10000
```

### Test with Desktop Limit
```bash
node tools/assets/validate-manifest.mjs assets/3d/luxury/ASSET_MANIFEST.json --maxTris=500000
```

### Test Specific Manifest
```bash
node tools/assets/validate-manifest.mjs path/to/custom/manifest.json --maxTris=75000
```

## 📈 Performance Guidelines

| Triangle Count | Mobile Performance | Battery Impact | Load Time |
|---------------|-------------------|----------------|-----------|
| < 10k | Excellent (60 FPS) | Minimal | < 1s |
| 10k - 50k | Good (30-60 FPS) | Low | 1-2s |
| 50k - 100k | Acceptable (30 FPS) | Moderate | 2-3s |
| 100k - 200k | Poor (15-30 FPS) | High | 3-5s |
| > 200k | Unusable | Very High | > 5s |

## 🔧 Extending the Validator

### Add Texture Size Check
```javascript
// In trianglesForFile() function
const textures = doc.getRoot().listTextures();
for (const texture in textures) {
  const size = texture.getSize();
  if (size > 4096) {
    warnings.push(`Texture exceeds 4K: ${size}px`);
  }
}
```

### Add File Size Check
```javascript
const stats = fs.statSync(optPath);
const sizeMB = stats.size / (1024 * 1024);
if (sizeMB > 10) {
  throw new Error(`File size ${sizeMB.toFixed(1)}MB exceeds 10MB limit`);
}
```

### Add Animation Check
```javascript
const animations = doc.getRoot().listAnimations();
if (animations.length > 0 && tris > 50000) {
  warnings.push(`Animated model with ${tris} triangles may impact performance`);
}
```

## 📝 Manifest Requirements

Each asset in `ASSET_MANIFEST.json` must have:

```json
{
  "id": "unique_identifier",          // Required: Unique ID
  "category": "category_name",        // Required: Asset category
  "opt": "path/to/optimized.glb",    // Required: Path to optimized model
  "license": "CC-BY-4.0",             // Required: License type
  "source": "sketchfab",              // Required: Source platform
  "source_url": "https://...",        // Required: Original URL
  "brandless": true,                  // Recommended: IP status
  "scale_m": 0.35,                    // Recommended: Real-world scale
  "polycount_hint": 25000             // Optional: Expected triangles
}
```

## 🚀 CI/CD Integration

### GitHub Actions Example
```yaml
- name: Validate 3D Assets
  run: |
    npm ci
    npm run validate:assets
```

### Pre-commit Hook
```bash
#!/bin/sh
node tools/assets/validate-manifest.mjs assets/3d/luxury/ASSET_MANIFEST.json --maxTris=100000
exit $?
```

## 📊 Metrics & Monitoring

The validator reports:
- Total assets validated
- Total triangle count across all assets
- Average triangles per asset
- Highest poly count asset
- Validation time

Use these metrics to track asset optimization over time.

---

**Version**: 2.0  
**Max Triangles**: 100,000 (default)  
**Last Updated**: 2025-08-16