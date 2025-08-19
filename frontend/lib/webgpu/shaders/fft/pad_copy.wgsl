// pad_copy.wgsl - center a smaller RG32F complex field into a larger RG32F target.
// Out-of-bounds regions are zero-filled. Optional Hann window on the copied region.
//
// Bindings (@group(0)):
//  @binding(0) read  : texture_storage_2d<rg32float, read>   src
//  @binding(1) write : texture_storage_2d<rg32float, write>  dst
//  @binding(2) uniform Params
//
// Dispatch over dst size:  ceil(dstW/16) x ceil(dstH/16)

struct Params {
  srcW   : u32;
  srcH   : u32;
  dstW   : u32;
  dstH   : u32;
  center : u32; // 1=centered, 0=top-left
  hann   : u32; // 1=apply 2D Hann in source region, 0=off
  _pad0  : u32;
  _pad1  : u32;
};

@group(0) @binding(0) var srcTex : texture_storage_2d<rg32float, read>;
@group(0) @binding(1) var dstTex : texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> P : Params;

const PI : f32 = 3.14159265358979323846;

fn hann1d(i: u32, n: u32) -> f32 {
  if (n <= 1u) { return 1.0; }
  let x = f32(i) / f32(n - 1u);
  return 0.5 * (1.0 - cos(2.0 * PI * x));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= P.dstW || gid.y >= P.dstH) { return; }

  // Compute top-left insertion offset
  let offX = select(0u, (P.dstW - P.srcW) / 2u, P.center == 1u);
  let offY = select(0u, (P.dstH - P.srcH) / 2u, P.center == 1u);

  // Map this dst pixel to a src pixel (if any)
  if (gid.x < offX || gid.y < offY || gid.x >= offX + P.srcW || gid.y >= offY + P.srcH) {
    textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return;
  }

  let sx = gid.x - offX;
  let sy = gid.y - offY;

  let s = textureLoad(srcTex, vec2<i32>(i32(sx), i32(sy)));
  var v = s.xy;

  if (P.hann == 1u) {
    let wx = hann1d(sx, P.srcW);
    let wy = hann1d(sy, P.srcH);
    let w2d = wx * wy;
    v *= w2d;
  }

  textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(v.x, v.y, 0.0, 0.0));
}