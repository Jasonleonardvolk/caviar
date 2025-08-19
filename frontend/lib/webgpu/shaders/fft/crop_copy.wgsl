// crop_copy.wgsl - extract a centered subregion from a larger RG32F complex field.
// The output size (outW,outH) is the texture size of dst.
//
// Bindings (@group(0)):
//  @binding(0) read  : texture_storage_2d<rg32float, read>   src
//  @binding(1) write : texture_storage_2d<rg32float, write>  dst
//  @binding(2) uniform Params
//
// Dispatch over dst size:  ceil(outW/16) x ceil(outH/16)

struct Params {
  srcW   : u32;
  srcH   : u32;
  outW   : u32;
  outH   : u32;
  center : u32; // 1=centered (default), 0=top-left crop
  _pad0  : u32;
  _pad1  : u32;
  _pad2  : u32;
};

@group(0) @binding(0) var srcTex : texture_storage_2d<rg32float, read>;
@group(0) @binding(1) var dstTex : texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> P : Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= P.outW || gid.y >= P.outH) { return; }

  let offX = select(0u, (P.srcW - P.outW) / 2u, P.center == 1u);
  let offY = select(0u, (P.srcH - P.outH) / 2u, P.center == 1u);

  let sx = offX + gid.x;
  let sy = offY + gid.y;

  // Bounds guard (should be safe if out <= src)
  if (sx >= P.srcW || sy >= P.srcH) {
    textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(0.0, 0.0, 0.0, 0.0));
    return;
  }

  let s = textureLoad(srcTex, vec2<i32>(i32(sx), i32(sy)));
  textureStore(dstTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(s.x, s.y, 0.0, 0.0));
}