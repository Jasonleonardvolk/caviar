// ${IRIS_ROOT}\frontend\lib\webgpu\pipelines\parallaxComposite.basic.wgsl
struct Params {
  width  : u32,
  height : u32,
  layerCount : u32,
};
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var views : texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read_write> outImg : array<vec4f>;
@group(0) @binding(3) var<uniform> params : Params;

fn idx(x:u32,y:u32,w:u32)->u32 { return y*w + x; }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }
  let uv = (vec2f(gid.xy)+vec2f(0.5))/vec2f(f32(params.width), f32(params.height));
  var acc : vec3f = vec3f(0.0);
  var wsum : f32 = 0.0;
  for (var l:u32 = 0u; l < params.layerCount; l++) {
    let c = textureSample(views, samp, uv, i32(l)).rgb;
    let w = 1.0;
    acc += c*w; wsum += w;
  }
  let color = select(vec4f(0.0,0.0,0.0,1.0), vec4f(acc / wsum, 1.0), wsum > 0.0);
  outImg[idx(gid.x, gid.y, params.width)] = color;
}
