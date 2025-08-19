// ${IRIS_ROOT}\frontend\lib\webgpu\effects\motionBlend.wgsl
struct Params { width:u32, height:u32, alpha:f32, } // alpha ~ mblurStrength
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var curr : texture_2d<f32>;
@group(0) @binding(2) var hist : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outImg : array<vec4f>;
@group(0) @binding(4) var<uniform> params : Params;

fn idx(x:u32,y:u32,w:u32)->u32 { return y*w + x; }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x>=params.width || gid.y>=params.height) { return; }
  let wh = vec2f(f32(params.width), f32(params.height));
  let uv = (vec2f(gid.xy)+vec2f(0.5))/wh;

  let c = textureSample(curr, samp, uv).rgb;
  let h = textureSample(hist, samp, uv).rgb;

  // EMA blend, light trail
  let a = clamp(params.alpha, 0.0, 1.0);
  let col = mix(c, h, a);
  outImg[idx(gid.x, gid.y, params.width)] = vec4f(col,1.0);
}
