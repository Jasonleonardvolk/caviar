// ${IRIS_ROOT}\frontend\lib\webgpu\effects\dofLight.wgsl
struct Params { width:u32, height:u32, strength:f32, }
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var scene : texture_2d<f32>;
@group(0) @binding(2) var depth : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> outImg : array<vec4f>;
@group(0) @binding(4) var<uniform> params : Params;

fn idx(x:u32,y:u32,w:u32)->u32 { return y*w + x; }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x>=params.width || gid.y>=params.height) { return; }
  let wh = vec2f(f32(params.width), f32(params.height));
  let uv = (vec2f(gid.xy)+vec2f(0.5))/wh;

  // simple CoC from normalized linear depth (0 near -> 1 far)
  let z = textureSample(depth, samp, uv).r;
  let coc = clamp(abs(z - 0.5) * 2.0 * params.strength, 0.0, 1.0); // focus at mid-depth

  var acc = vec3f(0.0);
  var wsum = 0.0;
  let off = vec2f(1.0/wh.x, 1.0/wh.y) * coc * 3.0; // up to ~3px radius

  // 7 taps in a cross + center (fast)
  let weights = array<f32,7>(0.28, 0.16, 0.12, 0.08, 0.12, 0.16, 0.08);
  let taps = array<vec2f,7>(vec2f(0.0,0.0), vec2f(1.0,0.0), vec2f(-1.0,0.0),
                           vec2f(0.0,1.0), vec2f(0.0,-1.0), vec2f(1.0,1.0), vec2f(-1.0,-1.0));
  for (var i:i32=0; i<7; i++) {
    let c = textureSample(scene, samp, uv + taps[i]*off).rgb;
    let w = weights[i];
    acc += c * w; wsum += w;
  }
  let col = acc / max(1e-4, wsum);
  outImg[idx(gid.x, gid.y, params.width)] = vec4f(col,1.0);
}
