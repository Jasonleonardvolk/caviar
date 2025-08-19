// ${IRIS_ROOT}\frontend\lib\webgpu\pipelines\viewBlendOcclusion.wgsl
struct Params {
  width:u32, height:u32,
  viewIndex:f32,   // target fractional view index
  layerCount:u32,
  edgeThresh:f32,  // 0.03..0.12 typical
};

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var views : texture_2d_array<f32>;
@group(0) @binding(2) var<storage, read_write> outImg : array<vec4f>;
@group(0) @binding(3) var<uniform> params : Params;

fn idx(x:u32,y:u32,w:u32)->u32 { return y*w + x; }
fn luma(c:vec3f)->f32 { return dot(c, vec3f(0.299,0.587,0.114)); }

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x>=params.width || gid.y>=params.height) { return; }
  let uv = (vec2f(gid.xy)+vec2f(0.5))/vec2f(f32(params.width), f32(params.height));

  let vi = clamp(params.viewIndex, 0.0, f32(params.layerCount - 1u));
  let v0 = i32(floor(vi));
  let v1 = i32(min(f32(params.layerCount - 1u), ceil(vi)));
  let t  = clamp(vi - floor(vi), 0.0, 1.0);

  let c0 = textureSample(views, samp, uv, v0).rgb;
  let c1 = textureSample(views, samp, uv, v1).rgb;

  // tiny Sobel on each layer (approx) to detect silhouettes; gate cross-blend near edges
  let off = vec2f(1.0 / f32(params.width), 1.0 / f32(params.height));
  let l0 = luma(c0);
  let l0x = luma(textureSample(views, samp, uv + vec2f(off.x,0), v0).rgb) - luma(textureSample(views, samp, uv - vec2f(off.x,0), v0).rgb);
  let l0y = luma(textureSample(views, samp, uv + vec2f(0,off.y), v0).rgb) - luma(textureSample(views, samp, uv - vec2f(0,off.y), v0).rgb);
  let g0 = length(vec2f(l0x,l0y));

  let l1 = luma(c1);
  let l1x = luma(textureSample(views, samp, uv + vec2f(off.x,0), v1).rgb) - luma(textureSample(views, samp, uv - vec2f(off.x,0), v1).rgb);
  let l1y = luma(textureSample(views, samp, uv + vec2f(0,off.y), v1).rgb) - luma(textureSample(views, samp, uv - vec2f(0,off.y), v1).rgb);
  let g1 = length(vec2f(l1x,l1y));

  // If one layer is strongly edged and the other is not, bias towards the edged one to avoid ghost double edges
  let e = params.edgeThresh;
  var w0 = 1.0 - t;
  var w1 = t;
  if (g0 > e && g1 <= e) { w0 = 1.0; w1 = 0.0; }
  if (g1 > e && g0 <= e) { w0 = 0.0; w1 = 1.0; }

  let col = (w0*c0 + w1*c1) / max(1e-5, w0 + w1);
  outImg[idx(gid.x, gid.y, params.width)] = vec4f(col, 1.0);
}
