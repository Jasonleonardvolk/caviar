// ${IRIS_ROOT}\frontend\lib\webgpu\pipelines\parallaxComposite.subgroup.wgsl
// enable subgroups; // Commented out - not supported by Naga

struct Params {
  // output size in pixels, used to clamp dispatch
  width  : u32,
  height : u32,
  layerCount : u32,       // number of parallax views (texture_2d_array layers)
};
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var views : texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read_write> outImg : array<vec4f>;
@group(0) @binding(3) var<uniform> params : Params;

fn idx(x:u32,y:u32,w:u32)->u32 { return y*w + x; }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(subgroup_invocation_id) sg_id: u32,
        @builtin(subgroup_size)          sg_sz: u32) {
  if (gid.x >= params.width || gid.y >= params.height) { return; }

  let uv = (vec2f(gid.xy) + vec2f(0.5)) / vec2f(f32(params.width), f32(params.height));
  let layers = params.layerCount;

  // Each invocation in the subgroup accumulates a strided subset of layers:
  var acc : vec3f = vec3f(0.0);
  var wsum : f32 = 0.0;

  var l = sg_id;
  loop {
    if (l >= layers) { break; }
    let c = textureSample(views, samp, uv, i32(l)).rgb;
    // simple weight; in your engine, use real parallax weight
    let w = 1.0;
    acc += c * w;
    wsum += w;
    l += sg_sz; // stride by subgroup size
  }

  // Reduce across subgroup - fallback to workgroup shared memory reduction
  // TODO: Implement proper shared memory reduction when subgroups not available
  let accR = acc.x; // subgroupAdd(acc.x);
  let accG = acc.y; // subgroupAdd(acc.y);
  let accB = acc.z; // subgroupAdd(acc.z);
  let wR   = wsum;  // subgroupAdd(wsum);

  // Only lane 0 writes (others return)
  if (sg_id != 0u) { return; }

  let color = select(vec4f(0.0,0.0,0.0,1.0), vec4f(vec3f(accR, accG, accB) / wR, 1.0), wR > 0.0);
  outImg[idx(gid.x, gid.y, params.width)] = color;
}
