# 🎯 FINAL PUSH - 121 Warnings to 0!

## Current Status: ✅ 0 FAILED, ⚠️ 121 WARNINGS

### Breakdown:
- 2 vec3 warnings (avatarShader.wgsl)
- 119 array bounds warnings (14 files)

---

## Step 1: Fix vec3 warnings (2 warnings)
```bash
node tools/shaders/fixes/fix_avatar_vec3.mjs
npm run shaders:sync ; npm run shaders:gate:iphone
```
**Expected: 121 → 119 warnings**

---

## Step 2: Add clamp helper to all files
```bash
node tools/shaders/fixes/add_clamp_helper.mjs
```

---

## Step 3: Fix array bounds by file (119 warnings)

### Priority Order (fix highest impact first):

#### 1. wavefieldEncoder_optimized.wgsl (20 warnings)
```wgsl
// Replace all array[i] with:
shared_spatial_freqs[i] → shared_spatial_freqs[clamp_index_dyn(i, 256u)]
spatial_freqs[i] → spatial_freqs[clamp_index_dyn(i, arrayLength(&spatial_freqs))]
dispersion_factors[channel % 3u] → dispersion_factors[clamp_index_dyn(channel % 3u, 3u)]
```

#### 2. wavefieldEncoder.wgsl (20 warnings)
Same pattern as above

#### 3. velocityField.wgsl (17 warnings)
```wgsl
shared_wavefield[idx] → shared_wavefield[clamp_index_dyn(idx, 100u)]
particles[idx] → particles[clamp_index_dyn(idx, arrayLength(&particles))]
spatial_freqs[idx] → spatial_freqs[clamp_index_dyn(idx, arrayLength(&spatial_freqs))]
phases[i/4u] → phases[clamp_index_dyn(i/4u, arrayLength(&phases))]
```

#### 4. butterflyStage.wgsl (14 warnings)
```wgsl
shared_data[idx] → shared_data[clamp_index_dyn(idx, 256u)]
input[idx] → input[clamp_index_dyn(idx, arrayLength(&input))]
output[idx] → output[clamp_index_dyn(idx, arrayLength(&output))]
twiddles[idx] → twiddles[clamp_index_dyn(idx, arrayLength(&twiddles))]
```

#### 5. transpose.wgsl (13 warnings)
```wgsl
tile[idx] → tile[clamp_index_dyn(idx, TILE_SIZE)]
input[idx] → input[clamp_index_dyn(idx, arrayLength(&input))]
output[idx] → output[clamp_index_dyn(idx, arrayLength(&output))]
```

#### 6. phaseOcclusion.wgsl (9 warnings)
```wgsl
// For neighbor access (idx-1, idx+1, idx-w, idx+w):
let safe_idx = clamp_index_dyn(idx, arrayLength(&occlusion));
let prev = select(0u, safe_idx - 1u, safe_idx > 0u);
let next = clamp_index_dyn(safe_idx + 1u, arrayLength(&occlusion));
occlusion[prev], occlusion[safe_idx], occlusion[next]
```

#### 7. propagation.wgsl (5 warnings)
```wgsl
shared_transfer[idx] → shared_transfer[clamp_index_dyn(idx, 256u)]
wavelengths[i] → wavelengths[clamp_index_dyn(i, 3u)]
spectral_weights[i] → spectral_weights[clamp_index_dyn(i, 3u)]
```

#### 8. multiDepthWaveSynth.wgsl (4 warnings)
```wgsl
depths[i] → depths[clamp_index_dyn(i, MAX_LAYERS)]
inputWave[idx] → inputWave[clamp_index_dyn(idx, arrayLength(&inputWave))]
outputWave[idx] → outputWave[clamp_index_dyn(idx, arrayLength(&outputWave))]
```

#### 9. multiViewSynthesis.wgsl (3 warnings)
```wgsl
shared_field_r[idx] → shared_field_r[clamp_index_dyn(idx, 256u)]
shared_field_g[idx] → shared_field_g[clamp_index_dyn(idx, 256u)]
shared_field_b[idx] → shared_field_b[clamp_index_dyn(idx, 256u)]
```

#### 10. bitReversal.wgsl (3 warnings)
```wgsl
bit_reversal[i] → bit_reversal[clamp_index_dyn(i, arrayLength(&bit_reversal))]
input[idx] → input[clamp_index_dyn(idx, arrayLength(&input))]
output[idx] → output[clamp_index_dyn(idx, arrayLength(&output))]
```

#### 11. fftShift.wgsl (2 warnings)
```wgsl
input[batch * N2 + i] → input[clamp_index_dyn(batch * N2 + i, arrayLength(&input))]
output[batch * N2 + j] → output[clamp_index_dyn(batch * N2 + j, arrayLength(&output))]
```

#### 12. normalize.wgsl (2 warnings)
```wgsl
input[idx] → input[clamp_index_dyn(idx, arrayLength(&input))]
output[idx] → output[clamp_index_dyn(idx, arrayLength(&output))]
```

---

## Validation After Each File:
```bash
npm run shaders:sync ; npm run shaders:gate:iphone
```

## 🏁 Target: 0 FAILED, 0 WARNINGS, 17 PASSED!

---

## Quick Script to Check Progress:
```bash
npm run shaders:gate:iphone | findstr "Warnings:"
```
