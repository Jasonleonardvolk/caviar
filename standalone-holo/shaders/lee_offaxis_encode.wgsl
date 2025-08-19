// lee_offaxis_encode.wgsl - off-axis Lee amplitude hologram
// Input  : amp_in  (A in [0,1])
//          phi_in  (phi in [-pi, pi])
// Output : holo_out (amplitude-only mask in [0,1]) - set binary=1 for 0/1 DMD/LCD
// Carrier: H = bias + scale * A * cos(phi + 2pi(fx*u + fy*v))

struct Params {
  width  : u32;
  height : u32;
  binary : u32;  // 0=grayscale, 1=binary
  _pad0  : u32;  // align to 16B
  fx_cycles: f32; // cycles across width
  fy_cycles: f32; // cycles across height
  bias   : f32;   // typically 0.5
  scale  : f32;   // typically 0.5
};

@group(0) @binding(0) var<storage, read>        amp_in  : array<f32>;
@group(0) @binding(1) var<storage, read>        phi_in  : array<f32>;
@group(0) @binding(2) var<storage, read_write>  holo_out: array<f32>;
@group(0) @binding(3) var<uniform>              P       : Params;

const PI : f32 = 3.14159265358979323846;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.width || gid.y >= P.height) { return; }
  let idx : u32 = gid.y * P.width + gid.x;

  let A   : f32 = clamp(amp_in[idx], 0.0, 1.0);
  let phi : f32 = phi_in[idx];

  let u = f32(gid.x) / f32(P.width);
  let v = f32(gid.y) / f32(P.height);
  let carrier = 2.0 * PI * (P.fx_cycles * u + P.fy_cycles * v);

  var H = P.bias + P.scale * A * cos(phi + carrier);
  H = clamp(H, 0.0, 1.0);

  if (P.binary == 1u) {
    H = select(0.0, 1.0, H >= 0.5);
  }

  holo_out[idx] = H;
}