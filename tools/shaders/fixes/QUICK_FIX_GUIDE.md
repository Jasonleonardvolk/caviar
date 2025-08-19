# Quick Warning Fix Guide

## 🚀 Quick Wins (16 warnings gone in 2 minutes)

### Step 1: Fix vec3 alignment (3 warnings)
```bash
node tools/shaders/fixes/fix_vec3_quick.mjs
```

### Step 2: Fix const/let (13 warnings)  
```bash
node tools/shaders/fixes/fix_const_let_quick.mjs
```

### Step 3: Validate progress
```bash
npm run shaders:sync && npm run shaders:gate:iphone
```
**Expected: Down from 135 → 119 warnings**

## 📋 Manual Array Bounds Fixes (119 warnings)

### Add helpers first:
```bash
node tools/shaders/fixes/add_clamp_helper.mjs
```

### Then manually fix these high-impact files:

#### butterflyStage.wgsl (14 warnings)
```wgsl
// Workgroup arrays (use constant):
shared_data[idx] → shared_data[clamp_index_dyn(idx, 256u)]

// Storage arrays (use arrayLength):
input[idx] → input[clamp_index_dyn(idx, arrayLength(&input))]
output[idx] → output[clamp_index_dyn(idx, arrayLength(&output))]
twiddles[idx] → twiddles[clamp_index_dyn(idx, arrayLength(&twiddles))]
```

#### velocityField.wgsl (17 warnings)
```wgsl
// Workgroup array:
shared_wavefield[idx] → shared_wavefield[clamp_index_dyn(idx, 100u)]

// Storage arrays:
spatial_freqs[idx] → spatial_freqs[clamp_index_dyn(idx, arrayLength(&spatial_freqs))]
phases[idx] → phases[clamp_index_dyn(idx, arrayLength(&phases))]
particles[idx] → particles[clamp_index_dyn(idx, arrayLength(&particles))]
```

#### wavefieldEncoder.wgsl & wavefieldEncoder_optimized.wgsl (20 warnings each)
```wgsl
// Workgroup arrays:
shared_spatial_freqs[i] → shared_spatial_freqs[clamp_index_dyn(i, 256u)]
shared_phases[i] → shared_phases[clamp_index_dyn(i, 256u)]
shared_amplitudes[i] → shared_amplitudes[clamp_index_dyn(i, 256u)]

// Fixed-size array:
dispersion_factors[channel % 3u] → dispersion_factors[clamp_index_dyn(channel % 3u, 3u)]
```

#### transpose.wgsl (13 warnings) - ALREADY DONE AS EXAMPLE

## 🎯 Common Patterns

### Pattern 1: Sequential access (i, i+1, i+2, i+3)
```wgsl
// BEFORE:
arr[i], arr[i+1u], arr[i+2u], arr[i+3u]

// AFTER:
let base = clamp_index_dyn(i, arrayLength(&arr) - 3u);
arr[base], arr[base+1u], arr[base+2u], arr[base+3u]
```

### Pattern 2: Neighbor access (idx-1, idx, idx+1)
```wgsl
// BEFORE:
arr[idx-1u], arr[idx], arr[idx+1u]

// AFTER:
let safe_idx = clamp_index_dyn(idx, arrayLength(&arr));
let prev = select(0u, safe_idx - 1u, safe_idx > 0u);
let next = clamp_index_dyn(safe_idx + 1u, arrayLength(&arr));
arr[prev], arr[safe_idx], arr[next]
```

### Pattern 3: 2D to 1D indexing
```wgsl
// BEFORE:
let idx = y * width + x;
arr[idx]

// AFTER:
let idx = y * width + x;
arr[clamp_index_dyn(idx, arrayLength(&arr))]
```

## 📊 Files by Warning Count (fix these first)
1. wavefieldEncoder_optimized.wgsl - 20 warnings
2. wavefieldEncoder.wgsl - 20 warnings  
3. velocityField.wgsl - 17 warnings
4. butterflyStage.wgsl - 14 warnings
5. transpose.wgsl - 13 warnings ✅ (example done)
6. phaseOcclusion.wgsl - 9 warnings
7. topologicalOverlay.wgsl - 7 warnings

## 🏁 Final Validation
After fixing each file:
```bash
npm run shaders:sync && npm run shaders:gate:iphone
```

Target: **0 FAILED, 0 WARNINGS, 17 PASSED** 🎉
