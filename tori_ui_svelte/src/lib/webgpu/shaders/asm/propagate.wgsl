@group(0) @binding(0) var srcField : texture_2d<f32>;                // RG32F: (re, im, 0, 0)
@group(0) @binding(1) var dstField : texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> dims : vec2<u32>;

@compute @workgroup_size(16, 16, 1)
fn propagate(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }

  let p = vec2<i32>(i32(gid.x), i32(gid.y));
  let v = textureLoad(srcField, p, 0);      // vec4<f32>
  var c = vec2<f32>(v.x, v.y);

  // multiply by band-limited H(fx,fy) here (sample LUT/compute inline)

  textureStore(dstField, p, vec4<f32>(c.x, c.y, 0.0, 0.0));
}