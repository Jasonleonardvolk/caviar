// phase_only_encode.wgsl - Arrizon double-phase encoder
// Input  : amp_in  (A in [0,1])  length = W*H
//          phi_in  (phi in [-pi, pi]) length = W*H
// Output : phase_out (phase map in [0, 2pi)) length = (2W)*H
// Layout : 2x1 superpixel: [phi+theta | phi-theta], where theta = arccos(A)

struct Params {
  width : u32;
  height: u32;
  _pad0 : u32;  // ensure 16B std140-like alignment for uniform blocks
  _pad1 : u32;
};

@group(0) @binding(0) var<storage, read>        amp_in   : array<f32>;
@group(0) @binding(1) var<storage, read>        phi_in   : array<f32>;
@group(0) @binding(2) var<storage, read_write>  phase_out: array<f32>;
@group(0) @binding(3) var<uniform>              P        : Params;

const PI : f32 = 3.14159265358979323846;

fn wrap2pi(x: f32) -> f32 {
  let t = x % (2.0 * PI);
  return select(t + 2.0 * PI, t, t >= 0.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let idx : u32 = gid.y * P.width + gid.x;

  let A   : f32 = clamp(amp_in[idx], 0.0, 1.0);
  let phi : f32 = phi_in[idx];
  let theta = acos(A);

  let left_phase  = wrap2pi(phi + theta);
  let right_phase = wrap2pi(phi - theta);

  let outW = P.width * 2u;
  let outIdxLeft  = gid.y * outW + (gid.x * 2u + 0u);
  let outIdxRight = gid.y * outW + (gid.x * 2u + 1u);

  phase_out[outIdxLeft]  = left_phase;
  phase_out[outIdxRight] = right_phase;
}