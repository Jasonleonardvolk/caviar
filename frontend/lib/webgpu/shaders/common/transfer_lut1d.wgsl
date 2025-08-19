// transfer_lut1d.wgsl - Use 1D radial LUT for transfer functions
// Saves 100x memory by exploiting radial symmetry of H(λ,z; fx,fy)

struct TFUniforms {
  z: f32,               // Propagation distance
  invLambda: f32,       // 1/wavelength
  radiusScale: f32,     // Scale factor for radius normalization
  lutSize: f32,         // Number of samples in LUT
}

@group(0) @binding(0) var<uniform> tfu: TFUniforms;
@group(0) @binding(1) var tf_lut: texture_1d<f32>;
@group(0) @binding(2) var tf_sampler: sampler;

// Get transfer function value from 1D LUT
fn transfer_from_lut(kx: f32, ky: f32) -> vec2<f32> {
  // Compute radius in frequency space
  let r = length(vec2<f32>(kx, ky));
  
  // Normalize to [0, 1] for texture sampling
  let r_norm = min(r * tfu.radiusScale, 0.999);
  
  // Sample the 1D texture (phase value)
  let phase = textureSampleLevel(tf_lut, tf_sampler, r_norm, 0.0).r;
  
  // Check for evanescent cutoff (phase = 0 means evanescent)
  if (abs(phase) < 1e-6) {
    return vec2<f32>(0.0, 0.0); // Zero transfer for evanescent waves
  }
  
  // Convert phase to complex number
  return vec2<f32>(cos(phase), sin(phase));
}

// Alternative: Direct computation without LUT (for comparison/fallback)
fn transfer_direct(kx: f32, ky: f32) -> vec2<f32> {
  let k = 2.0 * 3.141592653589793 * tfu.invLambda;
  let kr2 = kx * kx + ky * ky;
  let k2 = k * k;
  
  // Check for evanescent waves
  if (kr2 > k2) {
    return vec2<f32>(0.0, 0.0);
  }
  
  // Propagating wave
  let kz = sqrt(k2 - kr2);
  let phase = kz * tfu.z;
  
  return vec2<f32>(cos(phase), sin(phase));
}

// Entry point for propagation with LUT
@group(1) @binding(0) var<storage, read> field_in: array<vec2<f32>>;
@group(1) @binding(1) var<storage, read_write> field_out: array<vec2<f32>>;
@group(1) @binding(2) var<uniform> prop_params: PropagationParams;

struct PropagationParams {
  width: u32,
  height: u32,
  pixel_size: f32,
  use_lut: u32,  // 1 = use LUT, 0 = direct computation
}

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

@compute @workgroup_size(8, 8, 1)
fn propagate_with_lut(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= prop_params.width || gid.y >= prop_params.height) { return; }
  
  let idx = gid.y * prop_params.width + gid.x;
  
  // Get frequency coordinates
  let fx = (f32(gid.x) / f32(prop_params.width) - 0.5);
  let fy = (f32(gid.y) / f32(prop_params.height) - 0.5);
  
  // Scale to physical frequencies
  let kx = fx / prop_params.pixel_size;
  let ky = fy / prop_params.pixel_size;
  
  // Get transfer function value
  let H = select(
    transfer_direct(kx, ky),
    transfer_from_lut(kx, ky),
    prop_params.use_lut == 1u
  );
  
  // Apply transfer function to field
  field_out[idx] = complex_mul(field_in[idx], H);
}
