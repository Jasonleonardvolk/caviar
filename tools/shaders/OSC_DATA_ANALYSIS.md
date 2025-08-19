# CRITICAL ANALYSIS: osc_data Uniform Buffer Array Stride Issues
# ================================================================

## THE PROBLEM - osc_data WILL FAIL VALIDATION

The `OscillatorData` struct is currently declared as:
```wgsl
@group(1) @binding(0) var<uniform> osc_data: OscillatorData;
```

With this structure:
```wgsl
struct OscillatorData {
    psi_phase: f32,              // offset 0, size 4
    phase_coherence: f32,        // offset 4, size 4
    coupling_strength: f32,      // offset 8, size 4
    dominant_frequency: f32,     // offset 12, size 4
    phases: array<f32, 32>,      // offset 16 - PROBLEM!
    spatial_freqs: array<vec2<f32>, 32>,  // PROBLEM!
    amplitudes: array<f32, 32>   // PROBLEM!
}
```

## WHY IT WILL FAIL

WGSL Uniform Buffer Requirements:
- Arrays in uniform buffers MUST have stride >= 16 bytes
- Each array element must be padded to 16 bytes minimum

Current violations:
1. `phases: array<f32, 32>` 
   - Element size: 4 bytes (f32)
   - Required stride: 16 bytes
   - VIOLATION: 4 < 16 ❌

2. `spatial_freqs: array<vec2<f32>, 32>`
   - Element size: 8 bytes (vec2<f32>)
   - Required stride: 16 bytes  
   - VIOLATION: 8 < 16 ❌

3. `amplitudes: array<f32, 32>`
   - Element size: 4 bytes (f32)
   - Required stride: 16 bytes
   - VIOLATION: 4 < 16 ❌

## VALIDATION ERROR YOU'LL SEE
```
Shader validation error: 
Entry point main uses struct OscillatorData with array member that violates uniform buffer layout rules.
Array element stride must be at least 16 bytes in uniform buffers.
```

## TWO SOLUTIONS

### Option A: Convert to Storage Buffer (RECOMMENDED)
Same fix as wavefield_params - avoids all stride issues:

**WGSL Change:**
```wgsl
// BEFORE:
@group(1) @binding(0) var<uniform> osc_data: OscillatorData;

// AFTER:
@group(1) @binding(0) var<storage, read> osc_data: OscillatorData;
```

**TypeScript Changes:**
```typescript
// Buffer creation
device.createBuffer({
  size: oscDataSize,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST  // was UNIFORM
});

// Bind group layout
{
  binding: 0,
  visibility: GPUShaderStage.COMPUTE,
  buffer: { type: 'read-only-storage' }  // was 'uniform'
}
```

### Option B: Pad Arrays to 16-byte Stride (NOT RECOMMENDED)
Would require changing the struct AND the TypeScript side:

```wgsl
// Would need to change to:
struct OscillatorData {
    psi_phase: f32,
    phase_coherence: f32,
    coupling_strength: f32,
    dominant_frequency: f32,
    @stride(16) phases: array<f32, 32>,  // Forces 16-byte stride
    @stride(16) spatial_freqs: array<vec2<f32>, 32>,
    @stride(16) amplitudes: array<f32, 32>
}
```

But this requires padding data on CPU side - complex and wasteful!

## PERFORMANCE IMPACT

Storage buffers vs Uniform buffers:
- Storage: Slightly slower reads on some GPUs (~5-10%)
- Uniform: Faster reads but strict alignment rules
- For this use case: Storage buffer overhead is negligible

## RECOMMENDATION

✅ **USE THE INCLUDED -IncludeOscData FLAG**

Run:
```powershell
.\Patch-Wavefield-Params-To-Storage.ps1 -Apply -IncludeOscData
```

This will fix BOTH wavefield_params AND osc_data in one go.

## FILES AFFECTED

Both osc_data declarations are in:
- `frontend\lib\webgpu\shaders\wavefieldEncoder.wgsl` (line 50)
- `frontend\public\hybrid\wgsl\wavefieldEncoder.wgsl` (line 50)

## TESTING AFTER FIX

After applying the storage buffer fix:
1. Shader validation errors should disappear
2. No changes needed to data layout
3. Performance should remain virtually identical
4. Arrays work without any stride padding

## THE AVALANCHE PATTERN

This is classic shader development:
1. Fix wavefield_params uniform → works
2. osc_data uniform fails next → same issue
3. Fix one, another appears
4. Eventually prop_params might fail too if it has arrays

**PROACTIVE FIX:** Convert ALL uniform buffers with arrays to storage buffers now!
