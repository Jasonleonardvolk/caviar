# Shader Validation & Management Tools

## 🎯 Purpose
Prevent shader validation failures and maintain clean, canonical shader sources.

## 📁 Directory Structure
```
tools/shaders/
├── guards/
│   └── check_uniform_arrays.mjs    # Guard against uniform buffer array violations
├── reports/                        # Validation reports (gitignored)
│   ├── shader_validation_*.json
│   ├── shader_validation_*.junit.xml
│   └── shader_validation_*_summary.txt
├── copy_canonical_to_public.mjs    # One-way sync canonical → public
├── validate_and_report.mjs         # Comprehensive validation wrapper
├── Fix-WGSL-Phase2.ps1            # Mechanical fixes (textureLoad, swizzle, etc)
├── Fix-WGSL-Phase3.ps1            # Alignment & reserved word fixes
├── shader_quality_gate_v2.mjs     # Core validator
└── device_limits.iphone15.json    # Metal/iOS device constraints
```

## 🔒 Source of Truth
- **CANONICAL**: `frontend/lib/webgpu/shaders/**/*.wgsl` (edit here)
- **BUILD OUTPUT**: `frontend/public/hybrid/wgsl/**/*.wgsl` (auto-generated, DO NOT EDIT)

## 🚀 Quick Start

### Setup (run once)
```bash
node tools/shaders/setup_package_scripts.mjs
```

### Daily Workflow
```bash
# Full validation check
npm run shaders:full-check

# Fix any issues
npm run shaders:fix-all        # Apply mechanical fixes
# Then manually fix any remaining issues

# Verify fixes
npm run shaders:gate
```

## 📋 Available Scripts

| Command | Description |
|---------|-------------|
| `npm run shaders:sync` | Copy canonical shaders to public directory |
| `npm run shaders:validate` | Basic validation |
| `npm run shaders:gate` | Full validation with device limits (CI uses this) |
| `npm run shaders:check-uniforms` | Check for uniform buffer array violations |
| `npm run shaders:fix-phase2` | Dry-run Phase 2 fixes (add `-Apply` to fix) |
| `npm run shaders:fix-phase3` | Dry-run Phase 3 fixes (add `-Apply` to fix) |
| `npm run shaders:fix-all` | Apply all mechanical fixes |
| `npm run shaders:full-check` | Complete validation pipeline |

## 🛡️ Guards & Gates

### Uniform Array Guard
Prevents `var<uniform>` with arrays that violate std140 16-byte stride:
```wgsl
// ❌ BAD - Will fail validation
var<uniform> data: array<f32, 32>;      // 4-byte stride
var<uniform> data: array<vec2<f32>, 32>; // 8-byte stride

// ✅ GOOD - Use storage buffers
var<storage, read> data: array<f32, 32>;
var<storage, read> data: array<vec2<f32>, 32>;
```

### Workgroup Size Limits
Metal/iOS limit: ≤256 total threads
```wgsl
// ❌ BAD
@workgroup_size(32, 32, 1)  // 1024 threads

// ✅ GOOD
@workgroup_size(16, 16, 1)  // 256 threads
@workgroup_size(8, 8, 4)    // 256 threads
```

## 🔄 CI/CD Integration

GitHub Actions workflow (`.github/workflows/shader-validate.yml`):
1. Checks for uniform array violations
2. Syncs canonical → public
3. Runs full validation with device limits
4. Uploads reports as artifacts
5. Posts summary to PRs

## 🚨 Common Issues & Fixes

### Uniform Buffer Array Stride
**Error**: `Uniform buffer array stride violation`
**Fix**: Convert to storage buffer
```typescript
// TypeScript
- usage: GPUBufferUsage.UNIFORM
+ usage: GPUBufferUsage.STORAGE

- buffer: { type: 'uniform' }
+ buffer: { type: 'read-only-storage' }
```

### Missing Mip Level
**Error**: `textureLoad requires mip level`
**Fix**: Run `npm run shaders:fix-phase2 -- -Apply`

### Workgroup Size Too Large
**Error**: `Workgroup size exceeds device limit`
**Fix**: Use `@workgroup_size(16,16,1)` or smaller

## 📊 Validation Reports

Reports are generated in `tools/shaders/reports/`:
- `shader_validation_<timestamp>.json` - Raw validation data
- `shader_validation_<timestamp>.junit.xml` - JUnit format for CI
- `shader_validation_<timestamp>_summary.txt` - Human-readable summary
- `shader_validation_latest.json` - Symlink to most recent

## 🎯 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All shaders passed |
| 1 | Warnings but no errors |
| 2 | At least one shader failed |
| 3 | Validator tool failed |

## 🔐 Preventing Regression

1. **Pre-commit hook** checks for uniform arrays
2. **Pre-build hook** syncs canonical → public
3. **CI validation** on every push/PR
4. **Canonical-only edits** prevent duplicate drift

## 📈 Migration from Chaos

### Before (Hell)
- Duplicate shader trees
- Manual edits in multiple places
- Uniform buffer arrays causing stride violations
- Line numbers drifting between copies
- Fixes "not sticking"

### After (Heaven)
- Single source of truth
- Automatic sync to build directory
- Storage buffers for arrays
- Consistent line numbers
- Enforced validation gates

## 🏷️ Version Tags

After successful validation:
```bash
git add -A
git commit -m "fix: shader validation passing"
git tag -a shaders-pass-YYYY-MM-DD -m "Description"
git push && git push --tags
```

## 📚 References

- [WebGPU Spec - Buffer Layouts](https://www.w3.org/TR/webgpu/#buffer-layouts)
- [WGSL Spec - Memory Layout](https://www.w3.org/TR/WGSL/#memory-layouts)
- [Metal Shading Language Spec](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [std140 Layout Rules](https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Memory_layout)

---
*Generated: 2025-08-08 | Layer 16 Victory 🎉*
