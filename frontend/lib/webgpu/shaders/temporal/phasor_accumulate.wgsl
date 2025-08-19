// phasor_accumulate.wgsl
// Temporal accumulation with proper phasor arithmetic and warping
// Each buffered field is reprojected to current view before accumulation

const PI : f32 = 3.141592653589793;

struct AccumParams {
  dx: f32,              // Current view displacement X
  dy: f32,              // Current view displacement Y  
  z: f32,               // Propagation distance
  invLambda: f32,       // 1/wavelength
  weight0: f32,         // Weight for current frame
  weight1: f32,         // Weight for previous frame 1
  weight2: f32,         // Weight for previous frame 2
  numBuffers: u32,      // Number of frames to accumulate (1-3)
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: AccumParams;
@group(0) @binding(1) var<storage, read> field_current: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> field_prev1: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> field_prev2: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> field_out: array<vec2<f32>>;

// Complex arithmetic functions
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

fn complex_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return a + b;
}

fn complex_scale(a: vec2<f32>, s: f32) -> vec2<f32> {
  return a * s;
}

fn complex_normalize(a: vec2<f32>) -> vec2<f32> {
  let mag = max(length(a), 1e-6);
  return a / mag;
}

// Apply reprojection with Fresnel correction (from reproject_kspace_fresnel.wgsl)
fn apply_tilt_with_fresnel(F: vec2<f32>, kx: f32, ky: f32,
                           dx: f32, dy: f32, z: f32, invLambda: f32) -> vec2<f32> {
  let phi_linear = 2.0 * PI * (kx*dx + ky*dy);
  let phi_fresnel = PI * invLambda * (dx*dx + dy*dy) / max(z, 1e-6);
  let use_quadratic = (dx*dx + dy*dy) > (0.01 * 0.01);
  let phi = select(phi_linear, phi_linear + phi_fresnel, use_quadratic);
  let p = vec2<f32>(cos(phi), sin(phi));
  return complex_mul(F, p);
}

@compute @workgroup_size(8, 8, 1)
fn accumulate_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  
  let idx = gid.y * params.width + gid.x;
  
  // Compute frequency coordinates for reprojection
  let fx = (f32(gid.x) / f32(params.width) - 0.5);
  let fy = (f32(gid.y) / f32(params.height) - 0.5);
  let pixel_size = 0.00001; // 10 microns
  let kx = fx / pixel_size;
  let ky = fy / pixel_size;
  
  // Current frame (no reprojection needed)
  var accumulated = complex_scale(field_current[idx], params.weight0);
  
  // Previous frame 1 (reproject by -dx, -dy to align with current)
  if (params.numBuffers >= 2u) {
    let prev1_warped = apply_tilt_with_fresnel(
      field_prev1[idx], kx, ky, 
      -params.dx, -params.dy,  // Negative to align with current view
      params.z, params.invLambda
    );
    accumulated = complex_add(accumulated, complex_scale(prev1_warped, params.weight1));
  }
  
  // Previous frame 2 (reproject by -2*dx, -2*dy to align with current)
  if (params.numBuffers >= 3u) {
    let prev2_warped = apply_tilt_with_fresnel(
      field_prev2[idx], kx, ky,
      -2.0 * params.dx, -2.0 * params.dy,  // Double displacement for older frame
      params.z, params.invLambda
    );
    accumulated = complex_add(accumulated, complex_scale(prev2_warped, params.weight2));
  }
  
  // Normalize to unit phasor to preserve phase relationships
  field_out[idx] = complex_normalize(accumulated);
}

// Alternative entry point for motion-aware accumulation
@compute @workgroup_size(8, 8, 1)
fn accumulate_with_motion(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  
  let idx = gid.y * params.width + gid.x;
  
  // Motion magnitude determines blend weights
  let motion_mag = length(vec2<f32>(params.dx, params.dy));
  let motion_factor = saturate(motion_mag * 10.0); // Scale to [0,1]
  
  // During fast motion, weight current frame more heavily
  let adjusted_w0 = mix(params.weight0, 0.9, motion_factor);
  let adjusted_w1 = mix(params.weight1, 0.08, motion_factor);
  let adjusted_w2 = mix(params.weight2, 0.02, motion_factor);
  
  // Renormalize weights
  let sum_w = adjusted_w0 + adjusted_w1 + adjusted_w2;
  let w0 = adjusted_w0 / sum_w;
  let w1 = adjusted_w1 / sum_w;
  let w2 = adjusted_w2 / sum_w;
  
  // Same accumulation logic with adjusted weights
  var accumulated = complex_scale(field_current[idx], w0);
  
  let fx = (f32(gid.x) / f32(params.width) - 0.5);
  let fy = (f32(gid.y) / f32(params.height) - 0.5);
  let pixel_size = 0.00001;
  let kx = fx / pixel_size;
  let ky = fy / pixel_size;
  
  if (params.numBuffers >= 2u) {
    let prev1_warped = apply_tilt_with_fresnel(
      field_prev1[idx], kx, ky, -params.dx, -params.dy,
      params.z, params.invLambda
    );
    accumulated = complex_add(accumulated, complex_scale(prev1_warped, w1));
  }
  
  if (params.numBuffers >= 3u) {
    let prev2_warped = apply_tilt_with_fresnel(
      field_prev2[idx], kx, ky, -2.0 * params.dx, -2.0 * params.dy,
      params.z, params.invLambda
    );
    accumulated = complex_add(accumulated, complex_scale(prev2_warped, w2));
  }
  
  field_out[idx] = complex_normalize(accumulated);
}

fn saturate(x: f32) -> f32 {
  return clamp(x, 0.0, 1.0);
}
