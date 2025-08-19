// ${IRIS_ROOT}\frontend\lib\webgpu\effects\chromaticAberration.wgsl
struct Params { width:u32, height:u32, strength:f32, }
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var scene : texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> outImg : array<vec4f>;
@group(0) @binding(3) var<uniform> params : Params;

fn idx(x:u32,y:u32,w:u32)->u32 { return y*w + x; }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x>=params.width || gid.y>=params.height) { return; }
  let wh = vec2f(f32(params.width), f32(params.height));
  let uv = (vec2f(gid.xy)+vec2f(0.5))/wh;
  let d = uv - vec2f(0.5,0.5);
  let r2 = dot(d,d);

  let s = params.strength * r2; // small at center, grows at edges
  let r = textureSample(scene, samp, uv + d * s).r;
  let g = textureSample(scene, samp, uv).g;
  let b = textureSample(scene, samp, uv - d * s).b;

  let col = vec3f(r,g,b);
  outImg[idx(gid.x, gid.y, params.width)] = vec4f(col, 1.0);
}
