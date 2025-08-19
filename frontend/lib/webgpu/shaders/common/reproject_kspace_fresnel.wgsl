// reproject_kspace_fresnel.wgsl
// Reprojection with correct Fresnel phase for small-to-moderate head motion
// Physics: phi_fresnel = π(Δx² + Δy²)/(λz), NOT πz(Δx² + Δy²)/λ

const PI : f32 = 3.141592653589793;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// kx, ky in cycles/m; dx, dy in meters; z in meters; invLambda = 1/λ (1/m)
fn apply_tilt_with_fresnel(F: vec2<f32>, kx: f32, ky: f32,
                           dx: f32, dy: f32, z: f32, invLambda: f32) -> vec2<f32> {
  let phi_linear   = 2.0 * PI * (kx*dx + ky*dy);
  
  // Correct quadratic: phi_fresnel = π * (dx²+dy²) / (λz) = PI * invLambda * (dx²+dy²) / z
  let phi_fresnel  = PI * invLambda * (dx*dx + dy*dy) / max(z, 1e-6);
  
  // Use quadratic correction for displacements > 1cm
  let use_quadratic = (dx*dx + dy*dy) > (0.01 * 0.01);
  let phi = select(phi_linear, phi_linear + phi_fresnel, use_quadratic);
  
  let p = vec2<f32>(cos(phi), sin(phi));
  return complex_mul(F, p);
}

// Entry point for reprojection compute pass
@group(0) @binding(0) var<storage, read> field_in: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> field_out: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> params: ReprojParams;

struct ReprojParams {
  dx: f32,
  dy: f32,
  z: f32,
  invLambda: f32,
  width: u32,
  height: u32,
  _pad0: u32,
  _pad1: u32,
}

@compute @workgroup_size(8, 8, 1)
fn reproject_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  
  let idx = gid.y * params.width + gid.x;
  let F = field_in[idx];
  
  // Compute frequency coordinates (normalized to [-0.5, 0.5])
  let fx = (f32(gid.x) / f32(params.width) - 0.5);
  let fy = (f32(gid.y) / f32(params.height) - 0.5);
  
  // Scale to physical frequencies (cycles/m)
  let pixel_size = 0.00001; // 10 microns, adjust as needed
  let kx = fx / pixel_size;
  let ky = fy / pixel_size;
  
  field_out[idx] = apply_tilt_with_fresnel(F, kx, ky, params.dx, params.dy, params.z, params.invLambda);
}
