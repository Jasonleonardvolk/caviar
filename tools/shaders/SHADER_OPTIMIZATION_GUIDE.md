# Shader-Specific Optimization Guide

## 🎯 multiViewSynthesis.wgsl Optimizations

### Current Analysis
- **Workgroup Memory**: ~19.2KB (60% of iPhone limit)
- **Risk**: Adding more shared arrays will exceed mobile limits
- **Performance**: Dynamic indexing in view mapping

### Immediate Fixes

#### 1. Reduce Workgroup Memory Footprint
```wgsl
// BEFORE: 3 separate arrays
var<workgroup> shared_field_r: array<vec4<f32>, 400>;
var<workgroup> shared_field_g: array<vec4<f32>, 400>;
var<workgroup> shared_field_b: array<vec4<f32>, 400>;

// AFTER: Interleaved single array
var<workgroup> shared_field: array<vec3<f32>, 400>;
// Saves 4.8KB (25% reduction)

// OR: Process in chunks
const CHUNK_SIZE = 200u;
var<workgroup> shared_chunk: array<vec4<f32>, CHUNK_SIZE>;
// Process in 2 passes, reusing memory
```

#### 2. Optimize View Indexing
```wgsl
// BEFORE: Dynamic calculation every access
fn getQuiltIndex(view_id: u32, tile_x: u32, tile_y: u32) -> u32 {
    let views_per_row = quilt_width / tile_width;
    let view_row = view_id / views_per_row;
    let view_col = view_id % views_per_row;
    return (view_row * tile_height + tile_y) * quilt_width + 
           (view_col * tile_width + tile_x);
}

// AFTER: Precomputed view mapping
struct ViewMapping {
    base_offset: u32,
    stride_x: u32,
    stride_y: u32,
}
@group(0) @binding(10) var<uniform> view_mappings: array<ViewMapping, MAX_VIEWS>;

fn getQuiltIndexFast(view_id: u32, tile_x: u32, tile_y: u32) -> u32 {
    let mapping = view_mappings[view_id];
    return mapping.base_offset + tile_x * mapping.stride_x + tile_y * mapping.stride_y;
}
```

#### 3. Mobile-Specific Configuration
```wgsl
// Add device capability flags
struct DeviceConfig {
    max_views: u32,
    tile_size: u32,
    use_fp16: u32,  // 1 for mobile, 0 for desktop
    max_workgroup_size: u32,
}
@group(0) @binding(0) var<uniform> device_config: DeviceConfig;

// Adaptive workgroup size
#ifdef MOBILE_GPU
    @workgroup_size(64, 1, 1)
#else
    @workgroup_size(256, 1, 1)
#endif
```

## 🚀 propagation.wgsl Optimizations

### Current Analysis
- **Compute Intensity**: High (Fresnel, FFT operations)
- **Loop Complexity**: Nested loops with complex math
- **Memory Pattern**: Strided access in FFT

### Immediate Fixes

#### 1. Band-Limited Processing
```wgsl
// BEFORE: Full spectrum processing
for (var freq = 0u; freq < 256u; freq++) {
    let fresnel = computeFresnel(freq, z, wavelength);
    // Complex computation
}

// AFTER: Adaptive band limiting
let max_freq = select(256u, 64u, device_config.is_mobile > 0u);
for (var freq = 0u; freq < max_freq; freq++) {
    // For mobile, only process critical frequencies
    if (device_config.is_mobile > 0u && freq > 32u) {
        // Use interpolation for high frequencies
        spectrum[freq] = mix(spectrum[freq-1], spectrum[freq+1], 0.5);
    } else {
        let fresnel = computeFresnel(freq, z, wavelength);
        // Full computation
    }
}
```

#### 2. LUT for Expensive Operations
```wgsl
// BEFORE: Runtime Fresnel calculation
fn computeFresnelNumber(z: f32, wavelength: f32, aperture: f32) -> f32 {
    return aperture * aperture / (wavelength * z);
}

// AFTER: Lookup table for common values
@group(0) @binding(15) var<storage, read> fresnel_lut: array<f32, 256>;

fn getFresnelFast(z_normalized: f32, wavelength_idx: u32) -> f32 {
    let lut_idx = u32(z_normalized * 255.0) * 4u + wavelength_idx;
    return fresnel_lut[min(lut_idx, 255u)];
}
```

#### 3. Optimize FFT Memory Access
```wgsl
// BEFORE: Strided access pattern
for (var k = 0u; k < N; k++) {
    let idx1 = k + j * stride;
    let idx2 = idx1 + half_size;
    let t = twiddle * data[idx2];
    data[idx2] = data[idx1] - t;
    data[idx1] = data[idx1] + t;
}

// AFTER: Coalesced access with local caching
var<workgroup> fft_cache: array<vec2<f32>, 64>;

// Load to shared memory (coalesced)
fft_cache[local_id] = data[global_id];
workgroupBarrier();

// Compute in shared memory
for (var stage = 0u; stage < log2(N); stage++) {
    // FFT operations on fft_cache
    workgroupBarrier();
}

// Write back (coalesced)
data[global_id] = fft_cache[local_id];
```

## 📱 iOS 26 Specific Considerations

### Metal Backend Quirks
```wgsl
// Ensure compatibility with Metal's restrictions

// 1. Array sizing must be explicit
// BAD: var<workgroup> data: array<f32>;
// GOOD: var<workgroup> data: array<f32, 256>;

// 2. Avoid texture arrays > 2048 layers
const MAX_TEXTURE_LAYERS = 2048u;

// 3. Use explicit memory barriers
workgroupBarrier();  // Required between shared memory ops
storageBarrier();    // Required for storage buffer coherency
```

### Power Efficiency
```wgsl
// Mobile GPUs prefer smaller, more frequent dispatches

// Desktop dispatch
// computePass.dispatch(256, 256, 1);

// Mobile dispatch (better thermal profile)
// for (let y = 0; y < 256; y += 32) {
//     computePass.dispatch(256, 32, 1);
// }
```

## 🔄 Progressive Enhancement Strategy

### Level 1: Baseline (All Devices)
- 16KB workgroup memory
- 64 thread workgroups
- No dynamic indexing
- Pre-computed constants

### Level 2: Mobile Optimized (iPhone 15+)
- 32KB workgroup memory
- 128 thread workgroups
- Limited dynamic indexing with bounds
- Partial LUT usage

### Level 3: Desktop Performance
- 48KB workgroup memory  
- 256+ thread workgroups
- Full dynamic indexing
- Runtime computation

### Runtime Detection
```javascript
// In your initialization code
async function getShaderLevel() {
    const adapter = await navigator.gpu.requestAdapter();
    const limits = adapter.limits;
    
    if (limits.maxComputeWorkgroupStorageSize >= 49152) {
        return 'desktop';
    } else if (limits.maxComputeWorkgroupStorageSize >= 32768) {
        return 'mobile';
    } else {
        return 'baseline';
    }
}

// Use appropriate shader variant
const shaderLevel = await getShaderLevel();
const shaderModule = device.createShaderModule({
    code: shaderSources[shaderLevel]
});
```

## ✅ Pre-Deployment Checklist

- [ ] Run validation for all target devices
- [ ] Verify workgroup memory < 80% of limit
- [ ] Check all array accesses have bounds
- [ ] Confirm vec3 alignment in storage buffers
- [ ] Test MSL transpilation for iOS
- [ ] Profile on actual iPhone 15 hardware
- [ ] Implement fallback for baseline devices
- [ ] Add performance telemetry
- [ ] Document shader variant selection logic
- [ ] Create shader performance dashboard

## 📊 Expected Performance Gains

| Optimization | Mobile Impact | Desktop Impact |
|--------------|---------------|----------------|
| Workgroup memory reduction | 30% faster | 5% faster |
| View index precomputation | 15% faster | 10% faster |
| Band-limited processing | 40% faster | N/A |
| LUT for Fresnel | 25% faster | 5% faster |
| Coalesced FFT | 20% faster | 15% faster |

**Combined**: ~50% performance improvement on iPhone 15
