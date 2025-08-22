# 🎯 Texture Validation Quick Reference

## ✅ Allowed Formats
```
PNG ✓ | JPG ✓ | JPEG ✓ | WebP ✓
```

## 📏 Size Limits

| Type | Max Size | Max Dimensions | File Size |
|------|----------|----------------|-----------|
| **Opaque** | 4096×4096 | 4K | 16MB |
| **Alpha** | 2048×2048 | 2K | 8MB |
| **Normal** | 2048×2048 | 2K | - |
| **Total** | - | - | 50MB |

## 🔍 Quick Checks

```bash
# Run validation
pnpm run validate:assets

# Test with stricter alpha
node tools/assets/validate-manifest.mjs assets/3d/luxury/ASSET_MANIFEST.json --maxAlphaDim=1024

# Test enhanced features
.\test-texture-validator.bat
```

## ⚠️ Common Errors

| Error | Solution |
|-------|----------|
| "texture format 'tiff' not allowed" | Convert to PNG/JPG |
| "alpha texture exceeds 2048px" | Resize to 2K or less |
| "normal map exceeds 2048px" | 2K is max for normals |
| "Total size 68MB exceeds 50MB" | Optimize/compress assets |

## 🎨 Optimization Tips

**Alpha Textures**
- Use PNG-8 instead of PNG-24
- Crop transparent areas
- Try WebP for better compression

**Normal Maps**
- 2K is sufficient for mobile
- Consider vertex colors for simple meshes

**Opaque Textures**
- JPG for photos (85% quality)
- PNG for UI/logos

## 🚀 Commands

```bash
# Validate
pnpm run validate:assets

# Test
node test-enhanced-validator.mjs

# Custom limits
--maxAlphaDim=1024 --maxAlphaBytes=4194304
```

---
**v3.0** | Alpha detection | Format allow-list | Normal limits