# Production Shader Quality Gate v2.0

## Overview
Production-grade WebGPU shader validation system with multi-backend support, device limit enforcement, and CI integration - essential for iOS 26 WebGPU deployment.

## 🚀 Quick Start

### Install Dependencies
```bash
# Install Naga validator (required)
cargo install naga-cli --version 26.0.0

# Install Node dependencies
npm install

# Optional: Install Tint for cross-compilation testing
# See: https://dawn.googlesource.com/dawn
```

### Run Validation

#### Windows (PowerShell)
```powershell
# Validate for iPhone 15
.\Validate-Shaders.ps1 -Target iphone15

# Validate and auto-fix issues
.\Validate-Shaders.ps1 -Target iphone15 -Fix

# Validate for all targets
.\Validate-Shaders.ps1 -Target all -Strict

# Install tools and validate
.\Validate-Shaders.ps1 -InstallTools -Target desktop
```

#### Windows (Batch)
```batch
REM Simple validation
validate_shaders.bat iphone15

REM Desktop validation
validate_shaders.bat desktop
```

#### Cross-platform (Node)
```bash
# Basic validation
node tools/shaders/shader_quality_gate_v2.mjs --dir=frontend/

# Full validation with all options
node tools/shaders/shader_quality_gate_v2.mjs \
  --dir=frontend/ \
  --strict \
  --fix \
  --targets=msl,hlsl,spirv \
  --limits=tools/shaders/device_limits.iphone15.json \
  --report=build/shader_report.json \
  --junit=build/shader_report.junit.xml
```

## 📱 Device Profiles

### Available Profiles
- **`iphone15`** - Conservative iOS limits (32KB workgroup memory)
- **`desktop`** - Desktop GPU limits (48KB workgroup memory)  
- **`webgpu-baseline`** - Minimum WebGPU spec (16KB workgroup memory)

### Key Limits Enforced

| Limit | iPhone 15 | Desktop | WebGPU Baseline |
|-------|-----------|---------|-----------------|
| Max Workgroup Invocations | 256 | 1024 | 256 |
| Max Workgroup Size X/Y | 256 | 1024 | 256 |
| Max Workgroup Size Z | 64 | 64 | 64 |
| Max Workgroup Storage | 32KB | 48KB | 16KB |
| Max Textures/Stage | 16 | 32 | 16 |
| Max Samplers/Stage | 16 | 16 | 16 |

## 🔍 Validation Rules

### Error-Level Checks
- **Duplicate Bindings** - No duplicate @group/@binding pairs
- **Workgroup Size Required** - All @compute functions need @workgroup_size
- **Workgroup Size Limits** - Dimensions within device capabilities
- **Workgroup Memory Limits** - Total shared memory within bounds
- **Naga Validation** - Syntax and semantic correctness
- **Cross-compilation** - Successful transpilation to MSL/HLSL

### Warning-Level Checks
- **Prefer Const** - Use const for compile-time values
- **Vec3 Alignment** - Proper padding in storage buffers
- **Dynamic Indexing** - Bounds checking for array access
- **Performance Hints** - Loop unrolling opportunities

## 🛠️ Auto-Fix Capabilities

Enable with `--fix` flag:
- Convert `let` to `const` for literals
- Add default `@workgroup_size(64, 1, 1)` 
- Insert vec3 padding fields
- Format consistency fixes

## 📊 Reports

### JSON Report (`build/shader_report.json`)
```json
{
  "timestamp": "2025-01-19T...",
  "summary": {
    "total": 15,
    "passed": 14,
    "failed": 1,
    "warnings": 3,
    "fixed": 2
  },
  "shaders": [
    {
      "file": "frontend/shaders/multiViewSynthesis.wgsl",
      "hash": "a1b2c3d4",
      "errors": [],
      "warnings": [
        {
          "rule": "WORKGROUP_MEMORY_LIMITS",
          "line": 45,
          "message": "Total workgroup memory (19200 bytes) approaching limit"
        }
      ],
      "backends": {
        "naga": { "success": true },
        "msl": { "success": true },
        "hlsl": { "success": true }
      }
    }
  ]
}
```

### JUnit XML (`build/shader_report.junit.xml`)
For CI integration - compatible with GitHub Actions, Jenkins, GitLab CI.

## 🔧 CI Integration

### GitHub Actions
The included workflow (`.github/workflows/shader-validation.yml`) runs on:
- Push to shader files
- Pull requests touching shaders
- Manual workflow dispatch

Features:
- Matrix builds for multiple device targets
- Artifact upload for reports
- PR comments with results
- JUnit test reporting

### Custom CI Setup
```yaml
# Example for other CI systems
shader-validation:
  script:
    - cargo install naga-cli --version 26.0.0
    - node tools/shaders/shader_quality_gate_v2.mjs --dir=frontend/ --strict
  artifacts:
    reports:
      - build/shader_report.*.json
      - build/shader_report.*.junit.xml
```

## 🎯 Shader Optimization Guidelines

### For `multiViewSynthesis.wgsl`
**Current**: ~19.2KB workgroup memory (3 arrays × 400 elements × vec4)

**Optimizations**:
```wgsl
// Before: Dynamic indexing
let tile_idx = view_id * TILES_PER_VIEW + local_tile;
shared_field_r[tile_idx] = data;

// After: Precomputed with bounds
let tile_idx = min(view_id * TILES_PER_VIEW + local_tile, 399u);
shared_field_r[tile_idx] = data;

// Consider: Tile data compression
// Instead of vec4<f32>, use vec2<f16> where precision allows
var<workgroup> shared_field_compressed: array<vec2<f16>, 800>;
```

### For `propagation.wgsl`
**Current**: Heavy compute with Fresnel/FFT operations

**Optimizations**:
```wgsl
// Before: Runtime Fresnel calculation
let fresnel = computeFresnelNumber(z, wavelength, aperture);

// After: LUT for common values
let fresnel_idx = getFresnelLUTIndex(z_normalized);
let fresnel = fresnel_lut[fresnel_idx];

// Mobile-specific: Band-limited processing
#ifdef MOBILE_GPU
  const MAX_FREQUENCY = 64u;  // Reduce from 128
#else
  const MAX_FREQUENCY = 128u;
#endif
```

### General Mobile Optimizations

1. **Workgroup Size Tuning**
```wgsl
// Desktop
@workgroup_size(256, 1, 1)

// Mobile (better occupancy)
@workgroup_size(64, 1, 1)
```

2. **Memory Access Patterns**
```wgsl
// Coalesced reads
let base_idx = workgroup_id.x * 256u;
for (var i = 0u; i < 4u; i++) {
  let idx = base_idx + local_id.x + i * 64u;
  data[i] = input_buffer[idx];
}
```

3. **Precision Optimization**
```wgsl
// Use f16 where possible on mobile
#ifdef SUPPORTS_F16
  var<workgroup> temp: array<f16, 512>;
#else
  var<workgroup> temp: array<f32, 256>;
#endif
```

## 📈 Performance Monitoring

### Capture Device Limits
```javascript
// Run in browser console on target device
const limits = await emitDeviceLimits('iphone15-pro');
// Saves device_limits.iphone15-pro.json
```

### Automated Testing
```bash
# Run on PR
node tools/shaders/shader_quality_gate_v2.mjs --targets=msl,hlsl

# Nightly comprehensive check
for target in iphone15 desktop webgpu-baseline; do
  node tools/shaders/shader_quality_gate_v2.mjs \
    --dir=frontend/ \
    --strict \
    --targets=msl,hlsl,spirv \
    --limits=tools/shaders/device_limits.$target.json
done
```

## 🚨 Common Issues & Solutions

### Issue: "Workgroup memory exceeds limit"
**Solution**: Reduce array sizes or use multiple passes
```wgsl
// Split into two passes
// Pass 1: Process first half
var<workgroup> shared_data: array<vec4<f32>, 200>;
// Pass 2: Process second half with same memory
```

### Issue: "Dynamic indexing without bounds"
**Solution**: Add explicit clamping
```wgsl
let safe_idx = min(dynamic_idx, array_size - 1u);
data[safe_idx] = value;
```

### Issue: "Vec3 alignment in storage"
**Solution**: Use vec4 or add padding
```wgsl
struct Vertex {
  position: vec3<f32>,
  _pad1: f32,  // Added by --fix
  normal: vec3<f32>,
  _pad2: f32,  // Added by --fix
}
```

## 🔮 Future Enhancements

- [ ] Performance profiling integration
- [ ] Shader complexity metrics
- [ ] Automatic device profile detection
- [ ] VSCode extension integration
- [ ] Shader minification for production
- [ ] Runtime validation hooks
- [ ] A/B testing framework for shader variants

## 📚 Resources

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [WGSL Specification](https://www.w3.org/TR/WGSL/)
- [Dawn/Tint Tools](https://dawn.googlesource.com/dawn)
- [Naga Validator](https://github.com/gfx-rs/naga)
- [iOS WebGPU Implementation](https://webkit.org/blog/webgpu)

## 📝 License

MIT License - See LICENSE file

---

**Need Help?** 
- Check `build/shader_report.json` for detailed error messages
- Run with `--fix` to auto-correct common issues
- Use `--strict` to catch all potential problems before production
